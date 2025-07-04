import re
import numpy as np
import cv2
from cnstd import CnStd
from cnocr import CnOcr
from ultralytics import YOLO
import torch


def rotar_imagen_cv2(img, angulo):
    """
    Rota una imagen con OpenCV alrededor del centro manteniendo el tamaño original.
    Args:
        img (np.ndarray): Imagen cargada con cv2.
        angulo (float): Ángulo de rotación en grados (positivo: antihorario).
    Returns:
        np.ndarray: Imagen rotada con tamaño original.
    """
    (h, w) = img.shape[:2]
    centro = (w // 2, h // 2)
    # Matriz de rotación
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    # Aplicar rotación sin cambiar tamaño
    img_rotada = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=0)
    return img_rotada


def recorte_desde_4_puntos(imagen, puntos, scale=1.0):
    """
    Recorta una región rectangular dada por 4 puntos, la alinea horizontalmente,
    la rota si está en vertical (para que quede a lo largo),
    y devuelve la imagen recortada, el ángulo original y el ángulo corregido.
    Args:
        imagen (np.ndarray): Imagen de entrada.
        puntos (list): Lista de 4 puntos [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
        scale (float): Escala del recorte.
    Returns:
        tuple: (imagen_recortada_horizontal, angulo_original, angulo_correccion)
            - angulo_original: ángulo antihorario respecto al eje X
            - angulo_correccion: 0 si no se rotó, 90 si se rotó para dejarlo horizontal
    """
    pts = np.array(puntos, dtype="float32")

    # Ordenar puntos: top-left, top-right, bottom-right, bottom-left
    def ordenar_puntos(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    rect = ordenar_puntos(pts)
    (tl, tr, br, bl) = rect
    # Calcular dimensiones
    ancho_sup = np.linalg.norm(tr - tl)
    ancho_inf = np.linalg.norm(br - bl)
    max_ancho = int(max(ancho_sup, ancho_inf) * scale)
    alto_izq = np.linalg.norm(bl - tl)
    alto_der = np.linalg.norm(br - tr)
    max_alto = int(max(alto_izq, alto_der) * scale)
    # Calcular ángulo original en sentido antihorario desde el eje X
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    angulo_rad = np.atan2(dy, dx)
    angulo_original = np.degrees(angulo_rad)
    # Crear matriz de destino para perspectiva (sin rotación)
    dst = np.array(
        [[0, 0], [max_ancho - 1, 0], [max_ancho - 1, max_alto - 1], [0, max_alto - 1]],
        dtype="float32",
    )
    # Transformar la imagen
    M = cv2.getPerspectiveTransform(rect, dst)
    imagen_recortada = cv2.warpPerspective(imagen, M, (max_ancho, max_alto))
    # Verificar si está en vertical: alto > ancho
    if imagen_recortada.shape[0] > imagen_recortada.shape[1]:
        # Rotar 90° sentido horario
        imagen_recortada = cv2.rotate(imagen_recortada, cv2.ROTATE_90_CLOCKWISE)
        angulo_correccion = 90
    else:
        angulo_correccion = 0
    return imagen_recortada, angulo_original, angulo_correccion


def equlizar_ycrcb(imgen):
    if imgen.ndim == 2:
        return cv2.equalizeHist(imgen)
    if imgen.ndim == 3:
        img_eq = cv2.cvtColor(imgen, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = img_eq.transpose(2, 0, 1)
        y = cv2.equalizeHist(y)
        img_eq = cv2.merge((y, cr, cb))
        return cv2.cvtColor(img_eq, cv2.COLOR_YCR_CB2RGB)


def equlizar_hsv(imgen):
    if imgen.ndim == 2:
        return cv2.equalizeHist(imgen)
    if imgen.ndim == 3:
        img_eq = cv2.cvtColor(imgen, cv2.COLOR_RGB2HSV)
        h, s, v = img_eq.transpose(2, 0, 1)
        v = cv2.equalizeHist(v)
        img_eq = cv2.merge((h, s, v))
        return cv2.cvtColor(img_eq, cv2.COLOR_HSV2RGB)


def rellenar_con_fondo(imagen, nuevo_ancho, nuevo_alto, color_fondo=(255, 255, 255)):
    """
    Rellena una imagen con un fondo de color blanco (o el color especificado) hasta alcanzar las dimensiones deseadas.
    Si la imagen es más grande que el fondo, la redimensiona para que quepa.

    Args:
        imagen (np.ndarray): Imagen original (puede ser en escala de grises o color).
        nuevo_ancho (int): Ancho deseado de salida.
        nuevo_alto (int): Alto deseado de salida.
        color_fondo (tuple): Color de fondo (por defecto blanco).

    Returns:
        np.ndarray: Imagen con fondo centrado del tamaño especificado.
    """
    alto_ori, ancho_ori = imagen.shape[:2]

    # Si la imagen es más grande que el fondo, redimensionar para que quepa
    scale = min(nuevo_ancho / ancho_ori, nuevo_alto / alto_ori, 1.0)
    if scale < 1.0:
        new_size = (int(ancho_ori * scale), int(alto_ori * scale))
        imagen = cv2.resize(imagen, new_size, interpolation=cv2.INTER_AREA)
        alto_ori, ancho_ori = imagen.shape[:2]

    # Crear fondo
    if len(imagen.shape) == 2:  # Escala de grises
        fondo = np.full((nuevo_alto, nuevo_ancho), color_fondo[0], dtype=np.uint8)
    else:  # Color
        fondo = np.full((nuevo_alto, nuevo_ancho, 3), color_fondo, dtype=np.uint8)

    # Calcular desplazamientos para centrar
    x_offset = (nuevo_ancho - ancho_ori) // 2
    y_offset = (nuevo_alto - alto_ori) // 2

    # Insertar la imagen original centrada
    fondo[y_offset : y_offset + alto_ori, x_offset : x_offset + ancho_ori] = imagen

    return fondo


class Id_identicator:
    """
    Clase para detectar y extraer número de cédula y nombres desde imágenes usando YOLO + OCR.

    Args:
        path_yolo (str): Ruta al modelo YOLO entrenado.
        device (str): 'cuda', 'cpu' o None para auto.
    """

    def __init__(self, path_yolo: str, device: str = None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.patron = r"\b\d{10}\b|\b\d{9}-\d\b"
        self.model_yolo = YOLO(path_yolo)
        self.std = CnStd(context=self.device)
        self.cn_ocr = CnOcr(context=self.device)

    def run(self, img: np.ndarray, conf: float = 0.8):
        self.img = img
        results = self.model_yolo(img, conf=conf, device=self.device)
        for result in results:
            self.types = [int(cls) for cls in result.obb.cls.int()]
            self.points = result.obb.xyxyxyxy.cpu().numpy()
            self.img_array = cv2.cvtColor(result.plot()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        if self.types:
            self._recortar_y_rotar()
            return self._ocr_identify()
        else:
            print("No se detectaron objetos.")
            return None, None

    def _recortar_y_rotar(self):
        self.recortado_rotado, ang_ori, ang_corr = [], [], []
        for point in self.points:
            img_crop, a_ori, a_corr = recorte_desde_4_puntos(self.img, point)
            self.recortado_rotado.append(img_crop)
            ang_ori.append(a_ori)
            ang_corr.append(a_corr)
        self.ori_prom = np.mean(ang_ori)
        self.corr_prom = np.mean(ang_corr)

    def _limpieza(self, texto):
        return texto.replace("-", "")

    def _ocr_identify(self):
        self.ocr_cedula = None
        self.ocr_nombres = []
        self.angulo_predict = 0

        # Ordenar por tipo (descendente)
        ordenado = sorted(
            zip(self.types, self.recortado_rotado), key=lambda x: x[0], reverse=True
        )
        self.types, self.recortado_rotado = zip(*ordenado)

        for tipo, img in zip(self.types, self.recortado_rotado):
            if tipo == 1:  # Tipo cédula
                texto = self.cn_ocr.ocr_for_single_line(img).get("text", "")
                cedulas = re.findall(self.patron, texto)
                if cedulas:
                    self.ocr_cedula = self._limpieza(cedulas[0])
                else:
                    # Reintentar con imagen rotada
                    self.angulo_predict = 180
                    img_rot = rotar_imagen_cv2(img, self.angulo_predict)
                    texto = self.cn_ocr.ocr_for_single_line(img_rot).get("text", "")
                    cedulas = re.findall(self.patron, texto)
                    if cedulas:
                        self.ocr_cedula = self._limpieza(cedulas[0])
            else:  # Nombres
                try:
                    img = rotar_imagen_cv2(img, self.angulo_predict)
                    box_infos = self.std.detect(img)
                    for box in box_infos["detected_texts"]:
                        texto = self.cn_ocr.ocr_for_single_line(box["cropped_img"])
                        self.ocr_nombres.append(texto)
                except Exception as e:
                    print(f"OCR falló: {e}")

        return self.ocr_cedula, self.ocr_nombres

    def recogntion(self):
        cedula, nombres = self.ocr_cedula, self.ocr_nombres
        self.docs = {}
        palabras_separadas = []
        _apellidos = []
        _nombres = []
        if cedula:
            self.docs["N_ID"] = cedula
        if nombres:
            for nombre in nombres:
                palabras_separadas.extend(nombre["text"].split())
            for i, nom in enumerate(palabras_separadas):
                if i < 2:
                    _apellidos.append(nom)
                else:
                    _nombres.append(nom)
        self.docs["Apellidos"]=" ".join(_apellidos)
        self.docs["Nombres"]=" ".join(_nombres)
        return self.docs

    def ocr_all(self):
        self.ocr_all_text = []
        angulo_total = self.angulo_predict + self.ori_prom + self.corr_prom
        img_rot = rotar_imagen_cv2(self.img, angulo_total)
        box_infos = self.std.detect(img_rot)
        for box in box_infos["detected_texts"]:
            texto = self.cn_ocr.ocr_for_single_line(box["cropped_img"])
            self.ocr_all_text.append(texto)
        return self.ocr_all_text, img_rot
