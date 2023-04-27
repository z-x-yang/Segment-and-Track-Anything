import cv2
import numpy as np 

def mask2bbox(mask):
    if len(np.where(mask > 0)[0]) == 0:
        print(f'not mask')
        return np.array([[0, 0], [0, 0]]).astype(np.int64)

    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)

    x0 = np.min(np.nonzero(x_)[0])
    x1 = np.max(np.nonzero(x_)[0])
    y0 = np.min(np.nonzero(y_)[0])
    y1 = np.max(np.nonzero(y_)[0])

    return np.array([[x0, y0], [x1, y1]]).astype(np.int64)

def draw_outline(mask, frame):
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

    return frame

def draw_points(points, modes, frame):
    neg_points = points[np.argwhere(modes==0)[:, 0]]
    pos_points = points[np.argwhere(modes==1)[:, 0]]

    for i in range(len(neg_points)):
        point = neg_points[i]
        cv2.circle(frame, (point[0], point[1]), 8, (255, 80, 80), -1)
    
    for i in range(len(pos_points)):
        point = pos_points[i]
        cv2.circle(frame, (point[0], point[1]), 8, (0, 153, 255), -1)

    return frame

if __name__ == '__main__':
    mask = cv2.imread('./debug/mask.jpg', cv2.IMREAD_GRAYSCALE)
    frame = cv2.imread('./debug/frame.jpg')
    draw_frame = draw_outline(mask, frame)
    
    cv2.imwrite('./debug/outline.jpg', draw_frame)

    # bbox = mask2bbox(mask)
    # draw_0 = cv2.rectangle(mask, bbox[0], bbox[1], (0, 0, 255))
    # cv2.imwrite('./debug/rect.png', draw_0)