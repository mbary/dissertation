import os
import numpy as np

import MapExtrackt

class ExtractFeatureMaps:

    def __init__(self, model, image_name ,display_every_n = 10):

        self.model = model
        self.display_every_n = display_every_n
        self.image_name = image_name
        self._extraction()

    
    def _extraction(self):

        fe = MapExtrackt.FeatureExtractor(self.model)
        fe.set_image(self.image_name)

        conv2d_layers_indexes = [i for i,x in enumerate(fe.layer_names) if x == "Conv2d"]

        desired_indexes = conv2d_layers_indexes[::self.display_every_n]

        all_images = []

        for index in desired_indexes:

            all_images.append(np.array(fe[index]))
        
        all_images_stacked = np.vstack(all_images)

        return all_images_stacked




