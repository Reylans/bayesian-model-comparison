"""
A. Benavoli, F. Mangili, G. Corani, M. Zaffalon, and F. Ruggeri. 2014.
A Bayesian Wilcoxon signed-rank test based on the dirichlet process.
In Proceedings of the 31st International Conference on International Conference on Machine Learning -
Volume 32 (ICML'14). JMLR.org, II–1026–II–1034. https://dl.acm.org/doi/10.5555/3044805.3045007

Paper URL: https://proceedings.mlr.press/v32/benavoli14.pdf
"""
import numpy as np
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from datetime import datetime
from scipy.stats import dirichlet
from bayesian.bayesian_test.AbstractBayesian import AbstractBayesian
from bayesian.utils.plotting import plot_simplex, plot_histogram, plot_posterior_predictive_check
from bayesian.bayesian_test.utils import print_result, posterior_predictive_check_metrics, calculate_statistics


def calculate_region_prob(s: float, col: str) -> float:
    """
    Calculate the probability for a given region.

    :param s: Sample row.
    :param col: Column name indicating the region ('left', 'rope', 'right').
    :return: Probability for the region.
    """
    max_values = s == np.max(s)
    return 1 / np.sum(max_values) if max_values[col] else 0


def calculate_probabilities(rope: tuple[float, float], posterior_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate the probabilities of being less than, within, and greater than the ROPE (Region of Practical Equivalence).

    :param rope: Tuple representing the ROPE interval.
    :param posterior_df: Posterior sample distribution.
    :return: A tuple with the probabilities of being less than, within, and greater than the ROPE.
    """
    # Calculate the probabilities for left, right, and rope regions
    left_prob = np.mean(posterior_df.apply(lambda row: calculate_region_prob(row, 'left'), axis=1))
    right_prob = np.mean(posterior_df.apply(lambda row: calculate_region_prob(row, 'right'), axis=1))
    rope_prob = np.mean(posterior_df.apply(lambda row: calculate_region_prob(row, 'rope'),
                                           axis=1)) if rope is not None else None
    return left_prob, rope_prob, right_prob


def rdirichlet(alpha: np.ndarray, size: int) -> np.ndarray:
    """
    Generate samples from a Dirichlet distribution.

    :param alpha: Parameter vector for the Dirichlet distribution.
    :param size: Number of samples to generate.
    :return: Array of samples from the Dirichlet distribution.
    """
    return dirichlet(alpha).rvs(size)


class BayesianWilcoxonSignedRankTest(AbstractBayesian):
    def __init__(self, y1: np.ndarray, y2: np.ndarray, rope: Optional[tuple[float, float]], s: float = 0.5,
                 z0: float = 0.0, seed: int = 42):
        """
        Initialize the Bayesian Wilcoxon Signed-Rank Test.

        :param y1: A 1D array (num_instances,) of first datapoints.
            This array represents the independently generated performance statistics of the first method being compared.
            Each element corresponds to a single instance.
        :param y2: A 1D array (num_instances,) of second datapoints.
            This array represents the independently generated performance statistics of the second method being compared.
            Each element corresponds to a single instance.
        :param rope: Region of Practical Equivalence (ROPE) as a tuple (min, max).
            This defines the range of values within which the differences between the two methods are considered
            practically equivalent. If the difference falls within this range, the two methods are deemed to have
            no significant difference in practical terms.
        :param s: Smoothing parameter for the Dirichlet distribution (must be ≥ 0). Default is 0.5.
            Determines the weight of the prior belief on the posterior distribution. Larger values give more weight
            to the prior, while smaller values give more weight to the observed data.
        :param z0: Pseudo-observation value for the prior (must be ≥ 0). Default is 0.
            Centers the prior belief around this value. Choose based on prior knowledge or use a neutral/expected value
            if no prior information is available.
        :param seed: Random seed for reproducibility. Default is 42.
            This parameter ensures that the random processes in the model are consistent across runs, allowing for
            reproducible results.
        """
        super(BayesianWilcoxonSignedRankTest, self).__init__(stan_file='', rope=rope, seed=seed)
        # Ensure the number of runs in both datasets match for a paired test
        n_runs1 = y1.shape[0]
        n_runs2 = y2.shape[0]
        assert n_runs1 == n_runs2, (f'The Wilcoxon signed-rank Test is a paired test. The number of runs in the first '
                                    f'dataset ({n_runs1}) must match the number of runs in the second dataset '
                                    f'({n_runs2}). Please verify and correct your input data.')

        # Ensure datasets do not contain NaN values
        assert not np.any(np.isnan(y1)), ('The dataset y1 contains NaN values. '
                                          'Please remove or handle these NaNs before proceeding.')
        assert not np.any(np.isnan(y2)), ('The dataset y2 contains NaN values. '
                                          'Please remove or handle these NaNs before proceeding.')

        # Ensure the correctness of model parameters

        self.y1 = y1
        self.y2 = y2
        self.s = s
        self.z0 = z0
        self.n_runs = n_runs1

        # Save weights for PPC check
        self.weights = None

    def _transform_data(self) -> dict:
        pass

    def _calculate_posterior(self, idx: int, weights: np.ndarray, left_matrix: np.ndarray, rope_matrix: np.ndarray,
                             right_matrix: np.ndarray) -> dict:
        """
        Helper function to calculate the posterior for a given index.

        :param idx: Index of the sample.
        :param weights: Weights from the Dirichlet distribution.
        :param left_matrix: Matrix indicating left region.
        :param rope_matrix: Matrix indicating ROPE region.
        :param right_matrix: Matrix indicating right region.
        :return: Dictionary with the left, rope, and right sums.
        """
        weights_matrix = np.outer(weights[idx], weights[idx])
        left_posterior = np.sum(left_matrix * weights_matrix)
        right_posterior = np.sum(right_matrix * weights_matrix)
        rope_posterior = np.sum(rope_matrix * weights_matrix) if self.rope is not None else None
        return dict(
            left=left_posterior,
            rope=rope_posterior,
            right=right_posterior,
        )

    def fit(self, iter_sampling: int = 50000, **kwargs):
        """
        Fit the Bayesian Wilcoxon Signed-Rank Test model.

        :param iter_sampling: Number of samples to generate. Default is 50000
        :param kwargs: Additional arguments (not used).
        :return: None
        """
        # Set seed for reproducibility
        np.random.seed(seed=self.seed)

        x = self.y1 - self.y2

        # Create the parameter vector for the sampling of the weights
        weights_dir_params = np.concatenate(([self.s], np.ones(len(x))))

        # Add the pseudo-observation due to the prior to the sample vector
        sample = np.concatenate(([self.z0], x))

        # Generate weights from the Dirichlet distribution
        weights = rdirichlet(weights_dir_params, iter_sampling)

        # Prepare to get the terms for all the pairs i, j
        x_matrix = np.add.outer(sample, sample)

        # Get the elements to be summed for each event
        if self.rope is not None:
            left_matrix = x_matrix < 2 * self.rope[0]
            right_matrix = x_matrix > 2 * self.rope[1]
            rope_matrix = (x_matrix >= 2 * self.rope[0]) & (x_matrix <= 2 * self.rope[1])
        else:
            right_matrix = x_matrix > 0
            left_matrix = x_matrix < 0
            rope_matrix = None

        # Calculate the posterior for each sample
        self._fit = [self._calculate_posterior(i, weights, left_matrix, rope_matrix, right_matrix) for
                     i in range(iter_sampling)]

        # Set weights for PPC check
        self.weights = weights

    def _posterior_predictive_check(self, directory_path: str, file_path: str,
                                    file_name: str = 'posterior_predictive_check', font_size: int = 12,
                                    save: bool = True) -> None:
        """
        This function performs posterior predictive checks and generates plots comparing the observed data
        to the posterior predictive distributions.

        :param directory_path: Path to the parent directory where the files are stored.
        :param file_path: Path where the plot should be saved.
        :param file_name: Name of the file to save the plot. Default is 'posterior_predictive_check'.
        :param font_size: Font size for the plot text elements. Default is 12.
        :param save: Whether to save the plot to file. Default is True.
        :return: None
        """
        print(f'{datetime.now().time().strftime("%H:%M:%S")} - INFO: Running posterior predictive check.')
        # Retrieve posterior predictive samples
        n_draws = self.weights.shape[0]
        x = self.y1 - self.y2
        n_samples = x.shape[0]

        # Generate posterior predictive samples
        x_rep = np.zeros((n_draws, n_samples))
        y1_rep = np.zeros((n_draws, n_samples))
        y2_rep = np.zeros((n_draws, n_samples))

        for i in range(n_draws):
            # Sample indices from the range including the pseudo-observation
            sample_idx = np.random.choice(range(n_samples + 1), size=n_samples, p=self.weights[i])
            # Generate synthetic observations
            x_synthetic = np.concatenate(([self.z0], x))[sample_idx]
            x_rep[i, :] = x_synthetic
            y1_rep[i, :] = np.concatenate(([self.z0], self.y2))[sample_idx] + x_synthetic
            y2_rep[i, :] = np.concatenate(([self.z0], self.y1))[sample_idx] - x_synthetic

        # Transpose to match the expected shape
        x_rep = x_rep.T
        y1_rep = y1_rep.T
        y2_rep = y2_rep.T

        # Calculate and print PPC Metrics
        metrics_x = [posterior_predictive_check_metrics(x, x_rep[:, i], ranks=False) for i in range(x_rep.shape[1])]
        metrics_y1 = [posterior_predictive_check_metrics(self.y1, y1_rep[:, i], ranks=False) for i in
                      range(y1_rep.shape[1])]
        metrics_y2 = [posterior_predictive_check_metrics(self.y2, y2_rep[:, i], ranks=False) for i in
                      range(y2_rep.shape[1])]

        means_x, std_devs_x = calculate_statistics(metrics_x)
        means_y1, std_devs_y1 = calculate_statistics(metrics_y1)
        means_y2, std_devs_y2 = calculate_statistics(metrics_y2)
        print('\nPosterior Predictive Check Metrics')
        print(f'x:\nMeans: {means_x}\nStdDevs: {std_devs_x}\n'
              f'y1:\nMeans: {means_y1}\nStdDevs: {std_devs_y1}\n'
              f'y2:\nMeans: {means_y2}\nStdDevs: {std_devs_y2}\n')

        # Reshape to match the expected format for ArviZ
        x_rep = x_rep.reshape((1, n_draws, n_samples))
        y1_rep = y1_rep.reshape((1, n_draws, n_samples))
        y2_rep = y2_rep.reshape((1, n_draws, n_samples))

        posterior_df = pd.DataFrame(self._fit)
        post_dict = {
            var: posterior_df[var].values
            for var in ('left', 'right')
            if var in posterior_df
        }
        if self.rope is not None:
            post_dict['rope'] = posterior_df['rope'].values

        self.inf_data = az.from_dict(
            posterior=post_dict,
            posterior_predictive=dict(x_rep=x_rep, y1_rep=y1_rep, y2_rep=y2_rep),
            observed_data=dict(x=x, y1=self.y1, y2=self.y2),
        )

        # Generate the PPC plot
        variables = ['x', 'y1', 'y2']
        plot_posterior_predictive_check(inf_data=self.inf_data, variables=variables, n_draws=n_draws, show_plt=not save,
                                        font_size=font_size, seed=self.seed)

        # Save the plot if requested
        if save:
            self.save_plot(directory_path=directory_path, file_path=file_path, file_name=file_name)
        else:
            plt.tight_layout()

        # Plot Posterior densities in the style of John K. Kruschke’s book.
        az.plot_posterior(self.inf_data)
        plt.show()

    def analyse(self, posterior_predictive_check: bool = True, plot: bool = True, save: bool = True, round_to: int = 4,
                directory_path: str = 'results', file_path: str = 'bayesian_wilcoxon_signed_rank_test',
                file_name: Optional[str] = None, **kwargs) -> dict:
        """
        Analyse the results using the Bayesian Wilcoxon Signed-Rank Test model.

        :param posterior_predictive_check: Whether to do a posterior predictive check. Default is True.
        :param plot: Whether to generate and display plots. Default is True.
        :param save: Whether to save the results and plots to files. Default is True.
        :param round_to: Number of decimal places to round the results to. Default is 4.
        :param directory_path: Path to the parent directory where the files are stored. Default is 'results'.
        :param file_path: Directory path to save the files. Default is 'bayesian_wilcoxon_signed_rank_test.
        :param file_name: Name of the file to save the results and plots. If None, a default name based on the current timestamp will be used.
        :param kwargs: Additional keyword arguments for customized analysis and plotting.
        :return: A dictionary containing the analysis results, including posterior probabilities and additional details.
        """
        # Perform PPC check
        if posterior_predictive_check:
            file_name_ppc = f'{self._execution_time}' if file_name is None else file_name
            self._posterior_predictive_check(directory_path, file_path, file_name=f'{file_name_ppc}_ppc', font_size=10,
                                             save=save)

        # Convert the fit results to a DataFrame
        posterior_df = pd.DataFrame(self._fit)

        # Calculate the probabilities
        left_prob, rope_prob, right_prob = calculate_probabilities(self.rope, posterior_df)

        # Store the posterior probabilities in a dictionary
        posterior_probs = dict(
            left_prob=left_prob,  # Probability for the left region
            rope_prob=rope_prob,  # Probability for the ROPE region
            right_prob=right_prob,  # Probability for the right region
        )

        # Define the parameters of the analysis
        parameters = dict(
            rope=self.rope,  # Region of Practical Equivalence (ROPE)
            s=self.s,  # Smoothing parameter for the Dirichlet distribution
            z0=self.z0,  # Pseudo-observation value for the prior
            iter_sampling=self.iter_sampling,  # Number of samples generated
            seed=self.seed  # Random seed for reproducibility
        )

        # Compile the results into a dictionary
        results = dict(
            method='Bayesian signed-rank test',  # Method used for the analysis
            parameters=parameters,  # Parameters used in the analysis
            posterior_probabilities=posterior_probs,  # Posterior probabilities for the left, right, and ROPE regions
            posterior=posterior_df  # DataFrame containing the posterior samples
        )

        # Save the results if requested
        if save:
            self.save_results(results, directory_path=directory_path, file_path=file_path, file_name=file_name)

        # Print the rounded results
        rounded_results = print_result(results, round_to=round_to)
        rounded_results['posterior']= rounded_results['posterior'].dropna(axis='columns')
        print(f'\nPosterior:\n{rounded_results["posterior"].mean().round(round_to)}')
        print(f'\nPosterior Probabilities:\n{rounded_results["posterior_probabilities"]}')

        # Generate plots if requested
        if plot:
            if self.rope is None or self.rope == (0, 0):
                plot_histogram(posterior=posterior_df, round_to=round_to, show_plt=False, **kwargs)
            else:
                plot_simplex(posterior=posterior_df, posterior_probabilities=posterior_probs, round_to=round_to,
                             show_plt=False, **kwargs)

            # Save the plot if requested
            if save:
                self.save_plot(directory_path=directory_path, file_path=file_path, file_name=file_name)
                plt.show()
            else:
                plt.tight_layout()
                plt.show()
        return results
