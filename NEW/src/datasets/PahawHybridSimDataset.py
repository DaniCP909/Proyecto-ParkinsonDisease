from torch.utils.data import Dataset
from utils.CustomMorphOps import fit_into_normalized_canvas
from utils.PatchesOps import patches_per_image, patch_generator

class PahawHybridSimDataset(Dataset):

    def __init__(self, data: list[tuple], device, transform=None, patch_w=200, stepsize=2):
        zip_tasks, zip_labels = zip(*data)
        self.tasks = list(zip_tasks)
        self.labels = list(zip_labels)

        self.transform = transform
        self.device = device

        self.patch_w = patch_w
        self.stepsize=stepsize

        self.max_w = float('-inf')
        self.max_h = float('-inf')
        for task in self.tasks:
            self.max_w = max(self.max_w, task.getHeight())
            self.max_h = max(self.max_h, task.getWidth())

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        stroke_img = self.tasks[idx].getImage()
        label = self.labels[idx]

        task = self.tasks[idx]

        if self.transform:
            stroke_img = self.transform(stroke_img)
        
        fit_img = fit_into_normalized_canvas(stroke_img, self.max_h, self.max_w)

        n_patches = patches_per_image(fit_img.shape[1], self.patch_w, self.stepsize)
        patches = patch_generator(
            image=fit_img,
            device=self.device,
            n_patches=n_patches,
            patch_height=fit_img.shape[0],
            patch_width=self.patch_w,
            stepsize=self.stepsize,
        )

        timestamp_patches = []
        altitude_patches = []
        azimuth_patches = []
        pressure_patches = []

        for patch in patches:
            print("hola")