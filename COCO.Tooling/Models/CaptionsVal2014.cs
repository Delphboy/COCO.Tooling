using System;
using System.Collections.Generic;
using System.Text;

namespace COCO.Tooling.Models
{
    public class CaptionsVal2014
    {
        public Info info { get; set; }
        public Image[] images { get; set; }
        public License[] licenses { get; set; }
        public CaptionAnnotation[] annotations { get; set; }
    }
}
