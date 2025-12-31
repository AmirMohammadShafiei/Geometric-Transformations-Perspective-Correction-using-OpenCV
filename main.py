import cv2
import numpy as np
import os

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    img = param

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])

        # draw point
        cv2.circle(img, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(img, str(len(points)), (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Select 4 Points", img)

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4,2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

def save(out_dir, name, img):
    path = os.path.join(out_dir, f"{name}.png")
    cv2.imwrite(path, img)
    print("[SAVED]", path)

def perspective_crop(img, pts, interp=cv2.INTER_CUBIC):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH), flags=interp)

    return warped, M

def translate(img, tx, ty, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    M = np.float32([[1,0,tx],[0,1,ty]])
    return cv2.warpAffine(img, M, (w, h), flags=interp)

def rotate(img, angle, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=interp)

def scale(img, sx, sy, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    M = np.float32([[sx,0,0],[0,sy,0]])
    newW, newH = int(w*sx), int(h*sy)
    return cv2.warpAffine(img, M, (newW, newH), flags=interp)

def shear(img, shx=0.2, shy=0.0, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    M = np.float32([[1, shx, 0],
                    [shy, 1,  0]])
    newW = int(w + abs(shx)*h)
    newH = int(h + abs(shy)*w)
    return cv2.warpAffine(img, M, (newW, newH), flags=interp)


def approximate_undistort(img, k1=-0.2, k2=0.0):
    h, w = img.shape[:2]
    fx, fy = w, w
    cx, cy = w/2, h/2
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
    dist = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
    return cv2.undistort(img, K, dist)

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    c = cv2.createCLAHE(3.0, (8,8))
    l2 = c.apply(l)
    out = cv2.merge([l2, a, b])
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

def sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def run(image_path, out_dir="outputs_geom"):
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found! Check the path.")

    save(out_dir, "00_original", img)

    clone = img.copy()
    cv2.namedWindow("Select 4 Points", cv2.WINDOW_NORMAL)
    cv2.imshow("Select 4 Points", clone)
    cv2.setMouseCallback("Select 4 Points", mouse_callback, clone)

    print("\n✅ Click 4 corners (clock frame corners recommended).")
    print("✅ Press ENTER after selecting 4 points.\n")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER
            break

    cv2.destroyAllWindows()

    if len(points) != 4:
        raise ValueError("You must select exactly 4 points!")

    interps = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4
    }

    for name, ip in interps.items():
        warped, _ = perspective_crop(img, points, interp=ip)
        save(out_dir, f"01_perspective_{name}", warped)

    roi, _ = perspective_crop(img, points, interp=cv2.INTER_LANCZOS4)
    save(out_dir, "02_roi_best", roi)

    und = approximate_undistort(roi, k1=-0.15)
    save(out_dir, "03_undistorted_roi", und)

    transforms = {
        "translate": lambda im, ip: translate(im, 50, 30, ip),
        "rotate":    lambda im, ip: rotate(im, 15, ip),
        "scale":     lambda im, ip: scale(im, 1.2, 1.2, ip),
        "shear":     lambda im, ip: shear(im, shx=0.25, shy=0.0, interp=ip),
    }

    for tname, func in transforms.items():
        for iname, ip in interps.items():
            out = func(und, ip)
            save(out_dir, f"04_{tname}_{iname}", out)

    enhanced = clahe(und)
    enhanced = denoise(enhanced)
    enhanced = sharpen(enhanced)
    save(out_dir, "05_extra_filters_clahe_denoise_sharpen", enhanced)

    print("\n✅ DONE! All outputs saved in:", out_dir)


if __name__ == "__main__":
    # Example path:
    # Windows: r"C:\Users\YOURNAME\Desktop\tower.jpg"
    # Mac: "/Users/YOURNAME/Desktop/tower.jpg"
    image_path = r"C:\Users\kavosh\Desktop\tower.jpg"
    run(image_path)
