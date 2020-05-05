using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using COCO.Tooling.Models;
using OpenCvSharp;

namespace COCO.Tooling
{
    public static class Vision
    {
        private static string[] _labels = {
            "person",
            "bicycle",
            "car",
            "motorbike",
            "aeroplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "sofa",
            "pottedplant",
            "bed",
            "diningtable",
            "toilet",
            "tvmonitor",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush"
        };

        private static string _configLocation = @"C:\Users\HPS19\Documents\Github\final-year-project\data\models\YOLOv3coco\yolov3.cfg";
        private static string _modelLocation = @"C:\Users\HPS19\Documents\Github\final-year-project\data\models\YOLOv3coco\yolov3.weights";

        private static float kConfidenceThreshold = 0.40f;
        private static float kScoreThreshold = 0.40f;
        private static float kNmsThreshold = 0.0f;

        public static async Task<ComparrisonData> DetectObjectsYoloCoco(Mat image)
        {
            var net = OpenCvSharp.Dnn.CvDnn.ReadNetFromDarknet(_configLocation, _modelLocation);
            var scaleFactor = 1 / 255.0;
            var size = new Size(416, 416);
            var mean = new Scalar(0, 0, 0);
            var swap_rb = false;
            var crop = false;
            var blob = OpenCvSharp.Dnn.CvDnn.BlobFromImage(image, scaleFactor, size, mean, swap_rb, crop);

            var outLayers = net.GetUnconnectedOutLayers();
            var layerNames = net.GetLayerNames();

            var names = new List<string>(outLayers.Length);

            for (int i = 0; i < outLayers.Length; ++i)
            {
                names.Add(layerNames[outLayers[i] - 1]);
            }

            // invoke forward propagation
            var netOutput = new List<Mat>();
            net.SetInput(blob);

            for (int i = 0; i < names.Count; i++)
            {
                netOutput.Add(net.Forward(names[i]));
            }

            var classIds = new List<int>();
            var confidences = new List<float>();
            var boxes = new List<Rect>();

            foreach (var i in netOutput)
            {
                unsafe
                {
                    float* data = (float*)i.Data;
                    for (int j = 0; j < i.Rows; ++j, data += i.Cols)
                    {
                        var scores = i.Row(j).ColRange(5, i.Cols);
                        Point classId;
                        double confidence;
                        // Can't pass null like in C++
                        double blank1;
                        Point blank2;

                        Cv2.MinMaxLoc(scores, out blank1, out confidence, out blank2, out classId);
                        if (confidence > kConfidenceThreshold)
                        {
                            Rect box;
                            int cx = (int)(data[0] * image.Cols);
                            int cy = (int)(data[1] * image.Rows);
                            box.Width = (int)(data[2] * image.Cols);
                            box.Height = (int)(data[3] * image.Rows);

                            box.X = cx - box.Width / 2;
                            box.Y = cy - box.Height / 2;

                            boxes.Add(box);
                            confidences.Add((float)confidence);
                            classIds.Add(classId.X);
                        }
                    }
                }
            }

            int[] indicies;
            OpenCvSharp.Dnn.CvDnn.NMSBoxes(boxes, confidences, kScoreThreshold, kNmsThreshold, out indicies);

            var compBoxes = new List<double>();
            var compAreas = new List<double>();

            foreach (var box in boxes)
            {
                compBoxes.Add(box.X);
                compBoxes.Add(box.Y);
                compBoxes.Add(box.Width);
                compBoxes.Add(box.Height);

                compAreas.Add(box.Width * box.Height);
            }

            var categories = new List<string>();
            foreach(var id in classIds)
            {
                categories.Add(_labels[id]);
            }

            var comp = new ComparrisonData()
            {
                Areas = compAreas.ToArray(),
                BoundingBoxes = compBoxes.ToArray(),
                CategoryIds = classIds.ToArray(),
                Categories = categories.ToArray()
            };

            return comp;
        }
    }
}
