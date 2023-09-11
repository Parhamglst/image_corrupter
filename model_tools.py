import torch
import torch.utils.data as data
from resnet import resnet50
from torch.backends import cudnn
from torchvision import datasets
from torchvision import transforms
import numpy as np
from image_utils import corruptions, perturbations
from video_loader import VideoFolder
from scipy.stats import rankdata


class Model_tools:
    def load_weights(self, path_to_model):
        """this function loads the weights of a resnet50 model

        Args:
            path_to_model (str): path to the model weights

        Returns:
            ResNet50 object: resnet50 model with the weights loaded
        """

        model = torch.load(path_to_model)

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using GPU")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        model.to(device)

        model.eval()
        self.model = model

    def eval_corruption(self, path_to_corrupted_dataset, batch_size, mean, std):
        """this function evaluates model on a corrupted dataset

        Args:
            model (ResNet50 object): resnet50 model
            path_to_corrupted_dataset (str): path to the corrupted dataset

        Returns:

        """
        self.model.eval()

        cudnn.benchmark = True

        # dataloader = data.DataLoader(datasets.ImageFolder(root = os.getcwd() + path_to_corrupted_dataset,
        #                                      transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])),
        #                 batch_size=batch_size, shuffle= False,pin_memory=True)

        error_rates = []
        for distortion_name in corruptions:
            rate = self.show_performance(
                distortion_name,
                path_to_corrupted_dataset,
                batch_size,
                mean,
                std,
            )
            error_rates.append(rate)
            print(
                "Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}".format(
                    distortion_name, 100 * rate
                )
            )

        print(
            "mCE (unnormalized by AlexNet errors) (%): {:.2f}".format(
                100 * np.mean(error_rates)
            )
        )

    def show_performance(
        self, distortion_name, path_to_corrupted_dataset, batch_size, mean, std
    ):
        """copied from @hendrycks github repo: https://github.com/hendrycks/robustness
        calculates the average error of a model on a given distortion

        Args:
            model (torch model): model to be evaluated
            distortion_name (str): name of the distortion
            path_to_perturbed_dataset (str): path to the perturbed dataset
            batch_size (int): batch size
            mean (float): mean of the dataset
            std (float): standard deviation of the dataset

        Returns:
            float: average error
        """
        errs = []

        for severity in range(1, 6):
            print(
                path_to_corrupted_dataset + "/" + distortion_name + "/" + str(severity)
            )
            distorted_dataset = datasets.ImageFolder(
                root=path_to_corrupted_dataset
                + "/"
                + distortion_name
                + "/"
                + str(severity),
                transform=transforms.Compose(
                    [
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                ),
            )

            distorted_dataset_loader = torch.utils.data.DataLoader(
                distorted_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
            )

            correct = 0
            for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
                data = torch.tensor(data.cuda())

                output = self.model(data)

                pred = output.argmax(dim=1)
                correct += pred.eq(target.cuda()).sum()

            errs.append(1 - 1.0 * correct.cpu() / len(distorted_dataset))

        print("\n=Average", tuple(errs))
        return np.mean(errs)

    def eval_pert(
        self, perturbation_name, path_to_perturbed_dataset, batch_size, mean, std
    ):
        """this function evaluates model on a perturbed dataset

        Args:
            model (torch model): model to be evaluated
            perturbation_name (str): name of the distortion
            path_to_perturbed_dataset (str): path to the perturbed dataset
            batch_size (int): batch size
            mean (float): mean of the dataset
            std (float): standard deviation of the dataset
        """
        self.model.eval()
        cudnn.benchmark = True

        loader = torch.utils.data.DataLoader(
            VideoFolder(
                root=path_to_perturbed_dataset + "/" + perturbation_name,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize(mean, std)]
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        predictions, ranks = [], []
        with torch.no_grad():
            for data, target in loader:
                num_vids = data.size(0)
                data = data.view(-1, 3, 224, 224).cuda()

                output = self.model(data)

                for vid in output.view(num_vids, -1, 200):
                    predictions.append(vid.argmax(1).to("cpu").numpy())
                    ranks.append(
                        [
                            np.uint16(rankdata(-frame, method="ordinal"))
                            for frame in vid.to("cpu").numpy()
                        ]
                    )

        ranks = np.asarray(ranks)

        print("Computing Metrics\n")

        print(
            "Flipping Prob\t{:.5f}".format(
                self._flip_prob(predictions, perturbation_name)
            )
        )
        print(
            "Top5 Distance\t{:.5f}".format(
                self._ranking_dist(ranks, perturbation_name, mode="top5")
            )
        )
        print(
            "Zipf Distance\t{:.5f}".format(
                self._ranking_dist(ranks, perturbation_name, mode="zipf")
            )
        )

    def _flip_prob(self, predictions, perturbation, severity=1):
        noise_perturbation = True if "noise" in perturbation else False
        result = 0
        step_size = 1 if noise_perturbation else severity

        for vid_preds in predictions:
            result_for_vid = []

            for i in range(step_size):
                prev_pred = vid_preds[i]

                for pred in vid_preds[i::step_size][1:]:
                    result_for_vid.append(int(prev_pred != pred))
                    if not noise_perturbation:
                        prev_pred = pred

            result += np.mean(result_for_vid) / len(predictions)

        return result

    def _ranking_dist(self, ranks, perturbation, severity=1, mode="top5"):
        noise_perturbation = True if "noise" in perturbation else False
        result = 0
        step_size = 1 if noise_perturbation else severity

        for vid_ranks in ranks:
            result_for_vid = []

            for i in range(step_size):
                perm1 = vid_ranks[i]
                perm1_inv = np.argsort(perm1)

                for rank in vid_ranks[i::step_size][1:]:
                    perm2 = rank
                    result_for_vid.append(self._dist(perm2[perm1_inv], mode))
                    if not noise_perturbation:
                        perm1 = perm2
                        perm1_inv = np.argsort(perm1)

            result += np.mean(result_for_vid) / len(ranks)

        return result

    def _dist(self, sigma, mode="top5"):
        identity = np.asarray(range(1, 201))
        cum_sum_top5 = np.cumsum(np.asarray([0] + [1] * 5 + [0] * (199 - 5)))
        recip = 1.0 / identity
        if mode == "top5":
            return np.sum(np.abs(cum_sum_top5[:5] - cum_sum_top5[sigma - 1][:5]))
        elif mode == "zipf":
            return np.sum(np.abs(recip - recip[sigma - 1]) * recip)
