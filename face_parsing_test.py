import os
import cv2
import time
import numpy as np
import torch
from argparse import ArgumentParser
from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing import FaceParser as RTNetPredictor
from ibug.face_parsing.utils import label_colormap


def main() -> None:
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument(
        '--input', '-i', help='Input video path or webcam index (default=0)', default=0)
    parser.add_argument(
        '--output', '-o', help='Output file path', default=None)
    parser.add_argument('--fourcc', '-f', help='FourCC of the output video (default=mp4v)',
                        type=str, default='mp4v')
    parser.add_argument('--benchmark', '-b', help='Enable benchmark mode for CUDNN',
                        action='store_true', default=False)
    parser.add_argument('--no-display', help='No display if processing a video file',
                        action='store_true', default=False)
    parser.add_argument('--threshold', '-t', help='Detection threshold (default=0.8)',
                        type=float, default=0.8)
    parser.add_argument('--encoder', '-e', help='Method to use, can be either rtnet50 or rtnet101 (default=rtnet50)',
                        default='rtnet50') # choices=['rtnet50', 'rtnet101', 'resnet50'])

    parser.add_argument('--decoder', help='Method to use, can be either rtnet50 or rtnet101 (default=rtnet50)',
                        default='fcn', choices=['fcn', 'deeplabv3plus'])
    parser.add_argument('-n', '--num-classes', help='Face parsing classes (default=11)', type=int, default=11)
    parser.add_argument('--max-num-faces', help='Max number of faces',
                        default=50)
    parser.add_argument('--weights', '-w',
                        help='Weights to load, can be either resnet50 or mobilenet0.25 when using RetinaFace',
                        default=None)
    parser.add_argument('--device', '-d', help='Device to be used by the model (default=cuda:0)',
                        default='cuda:0')
    args = parser.parse_args()

    # Set benchmark mode flag for CUDNN
    torch.backends.cudnn.benchmark = args.benchmark
    # args.method = args.method.lower().strip()
    vid = None
    out_vid = None
    has_window = False
    face_detector = RetinaFacePredictor(threshold=args.threshold, device=args.device,
                                        model=(RetinaFacePredictor.get_model('mobilenet0.25')))
    face_parser = RTNetPredictor(
        device=args.device, ckpt=args.weights, encoder=args.encoder, decoder=args.decoder, num_classes=args.num_classes)

    colormap = label_colormap(args.num_classes)
    print('Face detector created using RetinaFace.')
    try:
        # Open the input video
        using_webcam = not os.path.exists(args.input)
        vid = cv2.VideoCapture(int(args.input) if using_webcam else args.input)
        assert vid.isOpened()
        alphas = np.linspace(0.75, 0.25, num=args.max_num_faces)
        if using_webcam:
            print(f'Webcam #{int(args.input)} opened.')
        else:
            print(f'Input video "{args.input}" opened.')

        # Open the output video (if a path is given)
        if args.output is not None:
            out_vid = cv2.VideoWriter(args.output, fps=vid.get(cv2.CAP_PROP_FPS),
                                      frameSize=(int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                 int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                      fourcc=cv2.VideoWriter_fourcc(*args.fourcc))
            assert out_vid.isOpened()

        # Process the frames
        frame_number = 0
        window_title = os.path.splitext(os.path.basename(__file__))[0]
        print('Processing started, press \'Q\' to quit.')
        while True:
            # Get a new frame
            _, frame = vid.read()
            if frame is None:
                break
            else:
                # Detect faces
                start_time = time.time()
                faces = face_detector(frame, rgb=False)
                elapsed_time = time.time() - start_time

                # Textural output
                print(f'Frame #{frame_number} processed in {elapsed_time * 1000.0:.04f} ms: ' +
                      f'{len(faces)} faces detected.')

                if len(faces) == 0:
                    continue
                # Parse faces
                start_time = time.time()
                masks = face_parser.predict_img(frame, faces, rgb=False)
                elapsed_time = time.time() - start_time

                # Textural output
                print(f'Frame #{frame_number} processed in {elapsed_time * 1000.0:.04f} ms: ' +
                      f'{len(masks)} faces parsed.')

                # # Rendering
                dst = frame
                for i, (face, mask) in enumerate(zip(faces, masks)):
                    bbox = face[:4].astype(int)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(
                        0, 0, 255), thickness=2)
                    alpha = alphas[i]
                    index = mask > 0
                    res = colormap[mask]
                    dst[index] = (1 - alpha) * frame[index].astype(float) + \
                        alpha * res[index].astype(float)
                dst = np.clip(dst.round(), 0, 255).astype(np.uint8)
                frame = dst
                # Write the frame to output video (if recording)
                if out_vid is not None:
                    out_vid.write(frame)

                # Display the frame
                if using_webcam or not args.no_display:
                    has_window = True
                    cv2.imshow(window_title, frame)
                    key = cv2.waitKey(1) % 2 ** 16
                    if key == ord('q') or key == ord('Q'):
                        print('\'Q\' pressed, we are done here.')
                        break
                frame_number += 1
    finally:
        if has_window:
            cv2.destroyAllWindows()
        if out_vid is not None:
            out_vid.release()
        if vid is not None:
            vid.release()
        print('All done.')


if __name__ == '__main__':
    main()
