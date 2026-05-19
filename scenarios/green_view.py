import numpy as np
import cv2
from inference.segmenter import CLASS_IDX


def evaluate_green_view(mask, original_image_bgr, grid_size=3):
    tree_mask = (mask == CLASS_IDX['Tree']).astype(np.uint8)
    h, w = mask.shape

    overall = float(tree_mask.sum()) / (h * w)

    cell_h, cell_w = h // grid_size, w // grid_size
    grid_scores = np.zeros((grid_size, grid_size), dtype=np.float32)
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i*cell_h, (i+1)*cell_h
            x1, x2 = j*cell_w, (j+1)*cell_w
            cell = tree_mask[y1:y2, x1:x2]
            grid_scores[i, j] = float(cell.sum()) / cell.size

    vis = original_image_bgr.copy()
    overlay = vis.copy()
    overlay[tree_mask == 1] = [0, 200, 0]
    vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)

    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i*cell_h, (i+1)*cell_h
            x1, x2 = j*cell_w, (j+1)*cell_w
            score = grid_scores[i, j]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
            color = (0,0,255) if score < 0.10 else \
                    (0,165,255) if score < 0.25 else (0,255,0)
            cv2.putText(vis, f"{score*100:.1f}%",
                        (x1+10, y1+30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

    if overall >= 0.25:   grade = 'A (优秀)'
    elif overall >= 0.15: grade = 'B (良好)'
    elif overall >= 0.05: grade = 'C (一般)'
    else:                 grade = 'D (较差)'

    return {
        'visualization': vis,
        'overall_ratio': round(overall, 4),
        'grid_scores': grid_scores.tolist(),
        'grade': grade,
    }