using COCO.Tooling.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace COCO.Tooling
{
    internal class Program
    {
        private static async Task Main()
        {
            {
                await Run();
            }
        }

        private static async Task Run()
        {
            var imageLocation = @"";
            var captionsVal2014Location = @"";
            var instancesVal2014Location = @"";
            var outputLocation = @"";


            var data = await GenerateComparisonData(instancesVal2014Location);

            var avgObjScore = 0.0;
            var avgBoundingScore = 0.0;
            var groundTruthCounter = 1;

            foreach (var groundTruth in data)
            {
                var img = OpenCvSharp.Cv2.ImRead(@$"{imageLocation}\{groundTruth.FileName}");
                var comparrison = await Vision.DetectObjectsYoloCoco(img);

                var objectDetectorScore = await CalculatePercentageObjectDetectionCorrect(groundTruth.Categories, comparrison.Categories);
                var areaAccuracy = await CalculateAreaAccuracy(groundTruth, comparrison);

                avgObjScore += objectDetectorScore;
                avgBoundingScore += areaAccuracy;

                await WriteToCsvFile(outputLocation, $"{objectDetectorScore},{areaAccuracy}\n");

                Console.WriteLine($"{groundTruthCounter}/{data.Count} | {groundTruth.FileName} | Object detection score: {objectDetectorScore} \t Bounding Score: {areaAccuracy}");
                groundTruthCounter++;
            }

            Console.WriteLine();
            Console.WriteLine($"Average Object Detection Score: {avgObjScore / data.Count}");
            Console.WriteLine($"Average Bounding Area Score: {avgBoundingScore / data.Count}");
        }

        private static async Task<double> CalculateAreaAccuracy(GroundTruth groundTruth, ComparrisonData comparison)
        {
            var sumGroundTruth = 0.0;
            foreach (var a in groundTruth.Areas)
            {
                sumGroundTruth += a;
            }

            var sumComparison = 0.0;
            foreach (var a in comparison.Areas)
            {
                sumComparison += a;
            }

            return Math.Abs((sumGroundTruth - sumComparison) / sumGroundTruth);
        }

        private static async Task<Dictionary<string, int>> CountOccurances(List<string> list)
        {
            var dic = new Dictionary<string, int>();
            foreach (var str in list)
            {
                if (!dic.ContainsKey(str))
                {
                    dic.Add(str, 1);
                }
                else
                {
                    dic[str]++;
                }
            }
            return dic;
        }

        private static async Task<double> CalculatePercentageObjectDetectionCorrect(string[] groundTruthLabels, string[] compLables)
        {
            var truth = groundTruthLabels.ToList();
            var truthKeys = new HashSet<string>(groundTruthLabels);
            var truthCount = await CountOccurances(truth);

            var comp = compLables.ToList();
            var compCount = await CountOccurances(comp);

            int correct = 0;

            foreach (var key in truthKeys)
            {
                var score = 0;
                try
                {
                    var possibleScore = truthCount[key];
                    var minimiser = Math.Abs(compCount[key] - truthCount[key]);
                    score = possibleScore - minimiser;
                }
                catch
                {
                    score = 0;
                }
                correct += score;
            }

            return correct / (double)groundTruthLabels.Length;
        }

        private static async Task<List<GroundTruth>> GenerateComparisonData(string instancesLoc)
        {
            var required = new List<GroundTruth>();
            using (var fs = File.OpenRead(instancesLoc))
            {
                var valData = await JsonSerializer.DeserializeAsync<InstancesVal2014>(fs);

                var imageIds = valData.images.Select(x => x.id).ToArray();
                var fileNames = valData.images.Select(x => x.file_name).ToArray();

                for (int i = 0; i < imageIds.Length; i++)
                {
                    Console.WriteLine($"Generating comparison data {i + 1} / {imageIds.Length}");
                    var id = imageIds[i];
                    var fileName = fileNames[i];
                    var bboxes = valData.annotations.Where(x => x.image_id == id).SelectMany(x => x.bbox).ToArray();
                    var areas = valData.annotations.Where(x => x.image_id == id).Select(x => x.area).ToArray();
                    var categoryIds = valData.annotations.Where(x => x.image_id == id).Select(x => x.category_id).ToArray();

                    var categories = new List<string>();
                    foreach (var cId in categoryIds)
                    {
                        categories.Add(valData.categories.Where(x => x.id == cId).Select(x => x.name).FirstOrDefault());
                    }

                    required.Add(new GroundTruth()
                    {
                        ImageId = id,
                        FileName = fileName,
                        BoundingBoxes = bboxes,
                        Areas = areas,
                        CategoryIds = categoryIds,
                        Categories = categories.ToArray()
                    });
                }
            }
            Console.WriteLine();
            return required;
        }

        private static async Task WriteToCsvFile(string filePath, string line)
        {
            byte[] encodedText = Encoding.Unicode.GetBytes(line);
            using (var fs = new FileStream(filePath, FileMode.Append, FileAccess.Write, FileShare.None, bufferSize: 4096, useAsync: true))
            {
                await fs.WriteAsync(encodedText, 0, encodedText.Length);
            }
        }

        private List<FileInfo> LoadImagesIntoList(string location)
        {
            var directoryInfo = new DirectoryInfo(location);
            var files = directoryInfo.GetFiles("*.jpg");

            return new List<FileInfo>(files);
        }

        private List<FileInfo> SelectRandomImages(List<FileInfo> dataset, int count)
        {
            var randomImages = new List<FileInfo>();

            for (int i = 0; i < count; i++)
            {
                Random rnd = new Random();
                int index = rnd.Next(0, dataset.Count);

                randomImages.Add(dataset[index]);
            }

            return randomImages;
        }

        private string GetCaptionFromImage(FileInfo image, CaptionsVal2014 valData)
        {
            var fileName = image.Name;
            var id = valData.images.Single(s => s.file_name == fileName).id;
            return valData.annotations.First(s => s.image_id == id).caption;
        }

        private void CopyFilesToNewLocation(List<FileInfo> filesToCopy, string newLocation)
        {
            foreach (var file in filesToCopy)
            {
                file.CopyTo($@"{newLocation}/{file.Name}");
            }
        }
    }
}
