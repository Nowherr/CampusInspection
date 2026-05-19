import numpy as np
import cv2
from inference.segmenter import CLASS_IDX


def detect_illegal_parking(mask, original_image_bgr,
                           overlap_threshold=0.05,
                           min_car_area=500,
                           dilate_kernel=5):
    car_mask = (mask == CLASS_IDX['Car']).astype(np.uint8)
    sidewalk_mask = (mask == CLASS_IDX['Sidewalk']).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        car_mask, connectivity=8)

    vis = original_image_bgr.copy()
    alerts = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_car_area:
            continue

        single_car = (labels == i).astype(np.uint8)
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        dilated_car = cv2.dilate(single_car, kernel)

        overlap = int((dilated_car & sidewalk_mask).sum())
        overlap_ratio = overlap / area

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        if overlap_ratio > overlap_threshold:
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 3)
            label = f"ILLEGAL {overlap_ratio*100:.1f}%"
            cv2.putText(vis, label, (x, max(y-10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            alerts.append({
                'type': 'illegal_parking',
                'bbox': [x, y, w, h],
                'car_area': int(area),
                'overlap_ratio': round(float(overlap_ratio), 4),
                'severity': 'high' if overlap_ratio > 0.3 else 'medium'
            })
        else:
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return {
        'visualization': vis,
        'alerts': alerts,
        'total_cars': int(num_labels - 1),
        'illegal_count': len(alerts),
    }