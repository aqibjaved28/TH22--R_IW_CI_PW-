import torch
import nibabel as nib

class RegOutputsLabels:
    def __init__(self, output_path, label_path):
        """
        Initializes the processor with paths to output and label NIfTI files.

        Args:
            output_path (str): Path to the output NIfTI file.
            label_path (str): Path to the label NIfTI file.
        """
        self.output_path = output_path
        self.label_path = label_path
        
        # Load tensors
        self.nii_img_tensor_output = self._load_nii_as_tensor(self.output_path)
        self.nii_img_tensor_label = self._load_nii_as_tensor(self.label_path)

    def _load_nii_as_tensor(self, path):
        """
        Loads a NIfTI file and converts it to a PyTorch tensor.

        Args:
            path (str): Path to the NIfTI file.

        Returns:
            torch.Tensor: The reshaped tensor.
        """
        nii_img = nib.load(path)
        nii_img_tensor = torch.from_numpy(nii_img.get_fdata())
        x, y, z = nii_img_tensor.shape
        return torch.reshape(nii_img_tensor, (1, 1, x, y, z))

    def compute_scaled_tensors(self, n=10):
        """
        Computes scaled tensors for outputs and labels based on random multipliers.

        Args:
            n (int): The maximum value for random multipliers (default is 10).

        Returns:
            tuple: A tuple containing scaled output and label tensors.
        """
        # Generate random multipliers for scaling
        RANGE_VALUE_L_alpha = torch.randint(1, n + 1, (1,)).item()
        RANGE_VALUE_L_beta = torch.randint(1, n + 1, (1,)).item()

        L_alpha = RANGE_VALUE_L_alpha
        L_beta = RANGE_VALUE_L_beta

        print(f"L_alpha: {L_alpha}, L_beta: {L_beta}")

        # Scale the tensors
        r_outputs = self.nii_img_tensor_output.clone().detach().requires_grad_(True).to("cuda:0") * L_alpha
        r_labels = self.nii_img_tensor_label.to("cuda:0") * L_beta

        return r_outputs, r_labels
