
"""
- Preview/Record 모드 (Space 전환, ESC 종료)
- 녹화 중 프리뷰 화면에 빨간 원(REC) 표시
- VideoCapture(0 또는 인덱스/URL) + VideoWriter
- 추가 기능: 코덱 선택(C), 타겟 FPS 조정(9/0), 필터(G/F/[ ]/;/'/T), 스냅샷(P)
"""
import cv2 as cv
import argparse
from datetime import datetime
import os
import numpy as np

# ---------------------- Utils ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="OpenCV Video Recorder (Preview/Record with non-recorded REC indicator)")
    p.add_argument("--source", type=str, default="0", help="카메라 인덱스 '0' 또는 스트림 URL(rtsp://...)")
    p.add_argument("--outdir", type=str, default="records", help="영상/스냅샷 저장 폴더")
    p.add_argument("--fps", type=float, default=30.0, help="카메라 FPS 미보고 시 사용할 기본 FPS")
    p.add_argument("--width", type=int, default=1280, help="강제 가로 해상도 (0=기본값 유지)")
    p.add_argument("--height", type=int, default=960, help="강제 세로 해상도 (0=기본값 유지)")
    return p.parse_args()

def try_int(s):
    try:
        return int(s)
    except ValueError:
        return s

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def timestamped_path(outdir, prefix, ext):
    ensure_dir(outdir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(outdir, f"{prefix}_{ts}.{ext}")

def put_text(img, text, org, scale=0.55, color=(255,255,255), thickness=1):
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv.LINE_AA)

def draw_rec_circle(preview_img):
    h, w = preview_img.shape[:2]
    cv.circle(preview_img, (w-40, 40), 12, (0,0,255), -1)
    put_text(preview_img, "REC", (w-100, 46), 0.7, (0,0,255), 2)

def overlay_help(preview_img, codec, fps, size, filters_desc):
    y = 24
    lines = [
        "[Space] Record   [ESC] Close   [C] Codec   [9/0] FPS -/+   [P] Snapshot",
        "[G] Gray   [F] Flip   [T] Timestamp   [R] Reset   [ [/ ] ] Brightness -/+   [ ; / ' ] Contrast -/+",
        f"Codec={codec}  TargetFPS={fps:.1f}  Size={size[0]}x{size[1]}  Filters: {filters_desc}",
    ]
    for line in lines:
        # shadow
        put_text(preview_img, line, (10, y), 0.55, (0,0,0), 3)
        put_text(preview_img, line, (10, y), 0.55, (255,255,255), 1)
        y += 20

def filters_to_str(f):
    parts = []
    if f['gray']: parts.append("Gray")
    if f['flip']: parts.append("Flip")
    if f['timestamp']: parts.append("Time")
    if abs(f['alpha']-1.0)>1e-3: parts.append(f"α={f['alpha']:.2f}")
    if abs(f['beta'])>1e-3: parts.append(f"β={f['beta']:.0f}")
    return ",".join(parts) if parts else "None"

def apply_filters(frame, f):
    # 녹화에 포함될 프레임에 적용되는 필터
    out = frame
    if f['flip']:
        out = cv.flip(out, 1)
    out = cv.convertScaleAbs(out, alpha=f['alpha'], beta=f['beta'])
    if f['gray']:
        out = cv.cvtColor(out, cv.COLOR_BGR2GRAY)
        out = cv.cvtColor(out, cv.COLOR_GRAY2BGR)
    if f['timestamp']:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        put_text(out, ts, (10, out.shape[0]-10), 0.6, (30,230,30), 2)
    return out

# ---------------------- Main ----------------------
def main():
    args = parse_args()
    source = try_int(args.source)

    cap = cv.VideoCapture(source, cv.CAP_ANY)
    if args.width>0:  cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height>0: cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # 네트워크/고지연 소스 완화

    if not cap.isOpened():
        raise SystemExit("ERROR: 카메라/스트림을 열 수 없습니다. --source 값을 확인하세요.")

    # 캡처 해상도/FPS
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) or 1080
    size = (w, h)
    cam_fps = cap.get(cv.CAP_PROP_FPS)
    if cam_fps is None or cam_fps <= 0:
        cam_fps = float(args.fps)
    target_fps = float(cam_fps)

    # 코덱 순환
    codecs = [("mp4v","mp4"), ("XVID","avi"), ("MJPG","avi"), ("H264","mp4")]
    codec_idx = 0

    writer = None
    recording = False

    # 필터 상태
    filters = dict(gray=False, flip=False, timestamp=False, alpha=1.0, beta=0.0)

    print("[INFO] Space=Record, ESC=Close, C=Codec, 9/0=FPS -/+, G/F/T/R Filter, [ ] Brightness, ; ' Contrast, P=Snapshot")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] 프레임 읽기 실패. 재시도...")
            cv.waitKey(10)
            continue

        # --- (1) 녹화용 프레임: 필터 적용  ---
        record_frame = apply_filters(frame.copy(), filters)

        # --- (2) 프리뷰용 프레임: 녹화용 프레임 복사 후 오버레이 ---
        preview = record_frame.copy()
        if recording:
            draw_rec_circle(preview)  # 프리뷰 전용, 파일에는 기록하지 않음
        codec_name, _ = codecs[codec_idx]
        overlay_help(preview, codec_name, target_fps, size, filters_to_str(filters))

        # --- (3) 화면 표시 & 파일 기록 ---
        cv.imshow("CV Video Recorder", preview)
        if recording and writer is not None:
            writer.write(record_frame)  # 프레임 저장

        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # Space: Record 토글
            recording = not recording
            if recording:
                fourcc, ext = codecs[codec_idx]
                out_path = timestamped_path(args.outdir, "record", ext)
                ensure_dir(args.outdir)
                vw = cv.VideoWriter_fourcc(*fourcc)
                writer = cv.VideoWriter(out_path, vw, target_fps, size)
                if not writer.isOpened():
                    print("[ERROR] VideoWriter 생성 실패. 다른 코덱(C)을 선택하세요.")
                    recording = False
                    writer = None
                else:
                    print(f"[INFO] Recording 시작: {out_path} ({fourcc}, {target_fps:.1f} FPS)")
            else:
                if writer is not None:
                    writer.release()
                    writer = None
                print("[INFO] Recording 중지")
        elif key in (ord('c'), ord('C')):
            codec_idx = (codec_idx + 1) % len(codecs)
            print(f"[INFO] 코덱 선택: {codecs[codec_idx][0]}")
        elif key == ord('9'):
            if recording:
                print("[INFO] 현재 녹화 중입니다. FPS는 새 녹화부터 적용됩니다.")
            target_fps = max(1.0, target_fps - 1.0)
            print(f"[INFO] Target FPS: {target_fps:.1f}")
        elif key == ord('0'):
            if recording:
                print("[INFO] 현재 녹화 중입니다. FPS는 새 녹화부터 적용됩니다.")
            target_fps = min(240.0, target_fps + 1.0)
            print(f"[INFO] Target FPS: {target_fps:.1f}")
        elif key in (ord('g'), ord('G')):
            filters['gray'] = not filters['gray']
        elif key in (ord('f'), ord('F')):
            filters['flip'] = not filters['flip']
        elif key in (ord('t'), ord('T')):
            filters['timestamp'] = not filters['timestamp']
        elif key in (ord('r'), ord('R')):
            filters.update(dict(gray=False, flip=False, timestamp=False, alpha=1.0, beta=0.0))
        elif key == ord('['):
            filters['beta'] = max(-127.0, filters['beta'] - 5.0)   # 밝기 -
        elif key == ord(']'):
            filters['beta'] = min(127.0, filters['beta'] + 5.0)    # 밝기 +
        elif key == ord(';'):
            filters['alpha'] = max(0.1, filters['alpha'] - 0.1)    # 대비 -
        elif key == ord("'"):
            filters['alpha'] = min(5.0, filters['alpha'] + 0.1)    # 대비 +
        elif key in (ord('p'), ord('P')):
            shot = timestamped_path(args.outdir, "shot", "png")
            ensure_dir(args.outdir)
            cv.imwrite(shot, preview)  # 프리뷰 저장 (오버레이 포함)
            print(f"[INFO] 스냅샷 저장: {shot}")

    if writer is not None:
        writer.release()
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

