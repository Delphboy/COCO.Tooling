using System;
using System.Collections.Generic;
using System.Text;

namespace COCO.Tooling.Models
{
    public class GroundTruth
    {
        public long ImageId { get; set; }
        public string FileName { get; set; }
        public double[] BoundingBoxes { get; set; }
        public double[] Areas { get; set; }
        public long[] CategoryIds { get; set; }
        public string[] Categories { get; set; }
        public override string ToString()
        {
            return $"Image {ImageId}, has {CategoryIds.Length} objects in it";
        }
    }
}
