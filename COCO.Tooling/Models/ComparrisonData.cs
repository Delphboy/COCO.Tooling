using System;
using System.Collections.Generic;
using System.Text;

namespace COCO.Tooling.Models
{
    public class ComparrisonData
    {
        public double[] BoundingBoxes { get; set; }
        public double[] Areas { get; set; }
        public int[] CategoryIds { get; set; }
        public string[] Categories { get; set; }
    }
}
