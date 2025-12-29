import os
import glob
import time
import logging
import numpy as np
from dpeva.feature.generator import DescriptorGenerator

class FeatureWorkflow:
    """
    Workflow for generating descriptors for a dataset using a pre-trained model.
    """
    
    def __init__(self, config):
        self.config = config
        self._setup_logger()
        
        self.datadir = config.get("datadir")
        self.modelpath = config.get("modelpath")
        self.format = config.get("format", "deepmd/npy")
        self.output_mode = config.get("output_mode", "structural") # 'atomic' or 'structural'
        
        self.savedir = config.get("savedir", f"desc-{os.path.basename(self.modelpath).split('.')[0]}-{os.path.basename(self.datadir)}")
        
        self.head = config.get("head", "OC20M")
        self.batch_size = config.get("batch_size", 1000)
        self.omp_threads = config.get("omp_threads", 24)

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

    def run(self):
        self.logger.info("Start Generating Descriptors")
        
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
            
        generator = DescriptorGenerator(
            model_path=self.modelpath,
            head=self.head,
            batch_size=self.batch_size,
            omp_threads=self.omp_threads
        )
        
        start_time = time.perf_counter()
        
        # Iterate over systems in datadir
        systems = sorted(glob.glob(f"{self.datadir}/*"))
        if not systems:
            self.logger.warning(f"No systems found in {self.datadir}")
            return

        for item in systems:
            key = os.path.basename(item)
            save_key = os.path.join(self.savedir, key)
            
            output_filename = "desc_stru.npy" if self.output_mode == "structural" else "desc.npy"
            output_path = os.path.join(save_key, output_filename)
            
            if os.path.exists(output_path):
                self.logger.info(f"Descriptors for {key} already exist, skip")
                continue
                
            self.logger.info(f"Generating descriptors for {key} system")
            
            try:
                desc = generator.compute_descriptors(item, self.format, self.output_mode)
                
                if not os.path.exists(save_key):
                    os.mkdir(save_key)
                    
                np.save(output_path, desc)
                self.logger.info(f"Descriptors for {key} saved to {output_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to generate descriptors for {key}: {e}")
                
        end_time = time.perf_counter()
        self.logger.info(f"All Done! Total time: {end_time - start_time:.2f} sec")
