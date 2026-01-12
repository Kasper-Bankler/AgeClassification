import torch
import torch.nn as nn
from torchview import draw_graph
from transfer_model import TransferAgeModel

# --- The Wrapper Class ---
# This class exists ONLY to trick the visualizer.
# It brings the classifier layers "up" to the surface so we can see them,
# while keeping the MobileNet features packed "down" in a container.
class VisualizableModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        
        # 1. Keep MobileNet Features wrapped as one big block
        # Since this is a "Sequential" container, the visualizer will treat it as a single box
        # if we limit the depth correctly.
        self.MobileNet_Features = original_model.net.features
        
        # 2. Define the transition layers explicitly
        # (MobileNet does this internally, but we make them visible modules here)
        self.Global_Pool = nn.AdaptiveAvgPool2d((1, 1))
        self.Flatten = nn.Flatten()
        
        # 3. UNPACK the classifier layers
        # instead of keeping them in a generic "Sequential" box, we assign them
        # to individual names so they show up as distinct steps in the diagram.
        classifier = original_model.net.classifier
        self.Dropout_1 = classifier[0]
        self.Linear_1 = classifier[1]
        self.ReLU = classifier[2]
        self.Dropout_2 = classifier[3]
        self.Linear_2 = classifier[4]

    def forward(self, x):
        # Flow data through the big block
        x = self.MobileNet_Features(x)
        
        # Flow through transition
        x = self.Global_Pool(x)
        x = self.Flatten(x)
        
        # Flow through our custom layers individually
        x = self.Dropout_1(x)
        x = self.Linear_1(x)
        x = self.ReLU(x)
        x = self.Dropout_2(x)
        x = self.Linear_2(x)
        return x

# --- Main Execution ---
if __name__ == "__main__":
    # Load your real model
    real_model = TransferAgeModel()
    
    # Wrap it in our visualization structure
    viz_model = VisualizableModel(real_model)

    # Generate Graph
    # depth=1 is now perfect because our DisplayModel puts the 
    # "Features Box" and the "Classifier Layers" at the same top-level depth.
    graph = draw_graph(viz_model, input_size=(1, 3, 224, 224), depth=1)
    
    graph.visual_graph.render("final_architecture", format="png")
    print("Diagram saved as final_architecture.png")