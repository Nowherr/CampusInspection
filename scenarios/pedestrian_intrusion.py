import numpy as np
import cv2
from inference.segmenter import CLASS_IDX


def detect_pedestrian_intrusion(mask, original_image_bgr,
                                min_pedestrian_area=200):
    ped_mask = (mask == CLASS_IDX['Pedestrian']).astype(np.uint8)
    road_mask = (mask == CLASS_IDX['Road']).astype(np.uint8)
    sidewalk_mask = (mask == CLASS_IDX['Sidewalk']).astype(np.uint8)

    # 半透明叠加道路区域（红色）
    vis = original_image_bgr.copy()
    overlay = vis.copy()
    overlay[road_mask == 1] = [0, 0, 200]
    vis = cv2.addWeighted(overlay, 0.3, vis, 0.7, 0)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ped_mask, connectivity=8)

    alerts = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_pedestrian_area:
            continue

        single = (labels == i).astype(np.uint8)
        cx, cy = int(centroids[i][0]), int(centroids[i][1])

        on_road = float((single & road_mask).sum()) / area
        on_sidewalk = float((single & sidewalk_mask).sum()) / area

        if on_road > 0.5:
            risk, color = 'high', (0, 0, 255)
        elif on_road > 0.2:
            risk, color = 'medium', (0, 165, 255)
        elif on_sidewalk > 0.5:
            risk, color = 'low', (0, 255, 0)
        else:
            risk, color = 'unknown', (200, 200, 200)

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        cv2.circle(vis, (cx, cy), 5, color, -1)
        cv2.putText(vis, f"RISK:{risk.upper()}", (x, max(y-10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if risk in ('medium', 'high'):
            alerts.append({
                'type': 'pedestrian_intrusion',
                'centroid': [cx, cy],
                'bbox': [x, y, w, h],
                'on_road_ratio': round(on_road, 4),
                'risk_level': risk,
            })

    return {
        'visualization': vis,
        'alerts': alerts,
        'total_pedestrians': int(num_labels - 1),
        'high_risk_count': sum(1 for a in alerts if a['risk_level'] == 'high'),
    }