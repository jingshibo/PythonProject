'''
    two different loss function for cGAN model training. The second one is more advanced and preferred
'''

##
import torch


## loss function
class LossFunction():
    def __init__(self, criterion):
        self.criterion = criterion

    # Compute the variance of blending_factor along the height and width dimensions for each class, treat each channel separately
    def getVarianceWithinClass(self, blending_factor, image_one_hot_labels):
        n_classes = image_one_hot_labels.shape[1]

        class_specific_variances = []  # List to store variance for each class (timepoint conditions)
        # Loop through each class to calculate class-specific variance
        for i in range(n_classes):
            class_mask = image_one_hot_labels[:, i, 0, 0].bool()
            class_blending_factor = blending_factor[class_mask]

            if class_blending_factor.numel() > 0:
                # Compute variance along height and width dimensions of each channel for all samples in the class
                class_variance = torch.var(class_blending_factor, dim=[0, 2, 3]).sum()  # Sum along channels to get the result
                class_specific_variances.append(class_variance)

        # Calculate the mean variance, or return 0 if the list is empty
        average_class_specific_variance = sum(class_specific_variances) / len(class_specific_variances) if class_specific_variances else 0.0
        return average_class_specific_variance

    # computes the average absolute difference between the disc's prediction values for fake and real images (for each class separately)
    def getConstructDiffWithinClass(self, image_one_hot_labels, disc_fake_pred, disc_real_pred):
        # Number of classes in the data
        n_classes = image_one_hot_labels.shape[1]

        # Initialize list to store the mean absolute difference for each class
        class_specific_diffs = []
        # Loop through each class (timepoint conditions) to calculate the class-specific mean absolute differences
        for i in range(n_classes):
            # Create a boolean mask for samples belonging to the i-th class
            class_mask = image_one_hot_labels[:, i, 0, 0].bool()
            # Use the mask to extract the discriminator's predictions for fake and real images of the i-th class
            class_fake_pred = disc_fake_pred[class_mask]
            class_real_pred = disc_real_pred[class_mask]

            # Check if there are samples for this class in the batch
            if class_fake_pred.numel() > 0 and class_real_pred.numel() > 0:
                # Compute the mean absolute difference for the i-th class
                class_mean_diff = torch.abs(torch.mean(class_fake_pred) - torch.mean(class_real_pred))
                class_specific_diffs.append(class_mean_diff)

        # Compute the overall average mean absolute difference, or set it to 0 if no classes are represented in the batch
        average_class_specific_diff = sum(class_specific_diffs) / len(class_specific_diffs) if class_specific_diffs else 0.0
        return average_class_specific_diff

    def get_gen_loss(self, fake, image_one_hot_labels, one_hot_labels, disc):
        fake_image_and_labels = torch.cat((fake.float(), image_one_hot_labels.float()), 1)
        # error if you didn't concatenate your labels to your image correctly
        disc_fake_pred = disc(fake_image_and_labels, one_hot_labels)
        gen_loss = self.criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        return gen_loss

    def get_disc_loss(self, fake, real, image_one_hot_labels, one_hot_labels, disc):
        fake_image_and_labels = torch.cat((fake.float(), image_one_hot_labels.float()), 1)
        real_image_and_labels = torch.cat((real.float(), image_one_hot_labels.float()), 1)
        disc_fake_pred = disc(fake_image_and_labels.detach(), one_hot_labels)
        disc_real_pred = disc(real_image_and_labels, one_hot_labels)

        disc_fake_loss = self.criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = self.criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        return disc_loss


## GRADED FUNCTION: get_gradient
class WGANloss():
    def __init__(self, c_lambda, var_weight, construct_weight, factor_1_weight, factor_2_weight, factor_3_weight):
        self.c_lambda = c_lambda  # the weight of the gradient penalty
        self.var_weight = var_weight  # the weight of blending factor variance
        self.construct_weight = construct_weight  # the weight of blending factor variance
        self.factor_1_weight = factor_1_weight  # the weight of blending factor 1 value
        self.factor_2_weight = factor_2_weight  # the weight of blending factor 2 value
        self.factor_3_weight = factor_3_weight  # the weight of blending factor 3 value

    # Return the gradient of the discic's scores with respect to mixes of real and fake images.
    def get_gradient(self, disc, real, fake, epsilon, one_hot_labels):
        # Get mixed images
        mixed_images = real * epsilon + fake * (1 - epsilon)
        # Calculate the discic's scores on the mixed images
        mixed_scores = disc(mixed_images, one_hot_labels)
        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            # Note: You need to take the gradient of outputs with respect to inputs.
            inputs=mixed_images,
            outputs=mixed_scores,
            # These other parameters have to do with the pytorch autograd engine works
            grad_outputs=torch.ones_like(mixed_scores), create_graph=True, retain_graph=True)[0]
        return gradient

    # Return the gradient penalty, given a gradient.
    def gradient_penalty(self, gradient):
        # Flatten the gradients so that each row captures one image
        gradient = gradient.view(len(gradient), -1)
        # Calculate the magnitude of every row
        gradient_norm = gradient.norm(2, dim=1)
        # Penalize the mean squared distance of the gradient norms from 1
        penalty = torch.mean((gradient_norm - 1) ** 2)
        return penalty

    # computes the average absolute difference between the disc's prediction values for fake and real images (for each class separately)
    def getConstructDiffWithinClass(self, image_one_hot_labels, fake, real):
        # Number of classes in the data
        n_classes = image_one_hot_labels.shape[1]

        # Initialize list to store the mean absolute difference for each class
        class_specific_diffs = []
        # Loop through each class (timepoint conditions) to calculate the class-specific mean absolute differences
        for i in range(n_classes):
            # Create a boolean mask for samples belonging to the i-th class
            class_mask = image_one_hot_labels[:, i, 0, 0].bool()
            # Use the mask to extract the discriminator's predictions for fake and real images of the i-th class
            class_fake_pred = fake[class_mask]
            class_real_pred = real[class_mask]

            # Check if there are samples for this class in the batch
            if class_fake_pred.numel() > 0 and class_real_pred.numel() > 0:
                # Compute the mean absolute difference for the i-th class
                class_mean_diff = torch.abs(torch.mean(class_fake_pred) - torch.mean(class_real_pred))
                class_specific_diffs.append(class_mean_diff)

        # Compute the overall average mean absolute difference, or set it to 0 if no classes are represented in the batch
        average_class_specific_diff = sum(class_specific_diffs) / len(class_specific_diffs) if class_specific_diffs else 0.0
        return average_class_specific_diff

    # Compute the variance of blending_factor along the height and width dimensions for each class, treat each channel  separately
    def getVarianceWithinClass(self, blending_factor, image_one_hot_labels):
        n_classes = image_one_hot_labels.shape[1]

        class_specific_variances = []  # List to store variance for each class (timepoint conditions)
        # Loop through each class to calculate class-specific variance
        for i in range(n_classes):
            class_mask = image_one_hot_labels[:, i, 0, 0].bool()
            class_blending_factor = blending_factor[class_mask]

            if class_blending_factor.numel() > 0:
                # Compute variance along height and width dimensions of each channel for all samples in the class
                if class_blending_factor.shape[3] == 5:  # only one electrode grid
                    class_variance = torch.var(class_blending_factor, dim=[0, 2, 3]).sum()  # Sum along channels to get the result
                elif class_blending_factor.shape[3] == 10:  # two electrode grids, you should compute the variance for two grids sepqrately
                    var_first_5_columns = torch.var(class_blending_factor[:, :, :, :5], dim=[0, 2, 3]).sum()
                    var_second_5_columns = torch.var(class_blending_factor[:, :, :, 5:], dim=[0, 2, 3]).sum()
                    class_variance = var_first_5_columns + var_second_5_columns
                else:
                    raise Exception('wrong data shape!')
                class_specific_variances.append(class_variance)

        # Calculate the mean variance, or return 0 if the list is empty
        average_class_specific_variance = sum(class_specific_variances) / len(class_specific_variances) if class_specific_variances else 0.0
        return average_class_specific_variance

    def getWeightedTerm(self, blending_factor):
        if blending_factor.shape[1] == 1:  # one alpha parameter
            weighted_term = self.factor_1_weight * blending_factor[:, 0, :, :]
        elif blending_factor.shape[1] == 2:  # two alpha parameters
            weighted_term = self.factor_1_weight * blending_factor[:, 0, :, :] + self.factor_2_weight * (1 - blending_factor[:, 1, :, :])
        elif blending_factor.shape[1] == 3:  # three alpha parameters
            weighted_term = self.factor_1_weight * blending_factor[:, 0, :, :] + self.factor_2_weight * (
                    1 - blending_factor[:, 1, :, :]) + self.factor_3_weight * blending_factor[:, 2, :, :]
        return weighted_term

    # return generator cost
    def get_gen_loss(self, fake, real, blending_factor, image_one_hot_labels, one_hot_labels, disc):
        fake_image_and_labels = torch.cat((fake.float(), image_one_hot_labels.float()), 1)
        real_image_and_labels = torch.cat((real.float(), image_one_hot_labels.float()), 1)
        disc_fake_pred = disc(fake_image_and_labels, one_hot_labels)
        disc_real_pred = disc(real_image_and_labels, one_hot_labels)

        # Compute the variance value of blending_factor along the height and width dimensions for each sample, treat each channel separately
        variance_value = self.getVarianceWithinClass(blending_factor, image_one_hot_labels)
        # Compute the construct error between the generated data and real data in order to minimize the difference per class
        construct_error = self.getConstructDiffWithinClass(image_one_hot_labels, fake, real)
        # weighted term blending factor 1 and blending factor 2
        weighted_term = self.getWeightedTerm(blending_factor)

        # Add the variance term and construct error to the loss
        gen_loss = -1. * torch.mean(disc_fake_pred) + self.var_weight * variance_value +\
                   self.construct_weight * construct_error + torch.mean(weighted_term)
        return gen_loss

    # Return the loss of a disc given the disc's scores for fake and real images, the gradient penalty, and gradient penalty weight.
    def get_disc_loss(self, fake, real, image_one_hot_labels, one_hot_labels, disc):
        fake_image_and_labels = torch.cat((fake.float(), image_one_hot_labels.float()), 1)
        real_image_and_labels = torch.cat((real.float(), image_one_hot_labels.float()), 1)
        disc_fake_pred = disc(fake_image_and_labels.detach(), one_hot_labels)
        disc_real_pred = disc(real_image_and_labels, one_hot_labels)

        epsilon = torch.rand(len(real), 1, 1, 1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), requires_grad=True)
        gradient = self.get_gradient(disc, real_image_and_labels, fake_image_and_labels.detach(), epsilon, one_hot_labels)
        g_p = self.gradient_penalty(gradient)

        disc_loss = torch.mean(disc_fake_pred) - torch.mean(disc_real_pred) + self.c_lambda * g_p  # torch.mean() averages all value in a tensor
        return disc_loss

