"""
Calvo, B., Shir, O. M., Ceberio, J., Doerr, C., Wang, H., Bäck, T., & Lozano, J. A. (2019, July).
Bayesian performance analysis for black-box optimization benchmarking. In Proceedings of the Genetic and Evolutionary
Computation Conference Companion (pp. 1789-1797). https://doi.org/10.1145/3319619.3326888

Paper URL: https://hal.science/hal-02179609/
Code URL: https://github.com/b0rxa/scmamp/blob/master/inst/stan/pl_model.stan .
"""
import warnings
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional
from datetime import datetime
from bayesian.bayesian_test.AbstractBayesian import AbstractBayesian
from bayesian.utils.plotting import plot_boxplot, plot_posterior_predictive_check
from bayesian.bayesian_test.utils import print_result, posterior_predictive_check_metrics, calculate_statistics


class BayesianPlackettLuceTest(AbstractBayesian):
    def __init__(self, x_matrix: np.ndarray, minimize: bool = True, prior: Optional[np.ndarray] = None,
                 algorithm_labels: Optional[list[str]] = None, seed: int = 42):
        """
        Initialize the Bayesian Plackett-Luce Test.

        :param x_matrix: A 2D array (num_instances, n_algorithms) of rankings or scores.
            This array represents the performance statistics for multiple algorithms across various problem instances.
            Each row corresponds to one problem instance, and each column corresponds to a different algorithm.
            Entries are the performance values for the algorithms on the respective problem instances.
        :param minimize: If True, lower values in x_matrix indicate better rankings.
            This boolean parameter specifies whether lower performance values are better. If True, the model interprets
            lower values as indicating better performance, which is essential for correctly computing the algorithm
            rankings.
        :param prior: Prior distribution for the model parameters.
            This optional parameter allows you to specify a prior distribution for the model parameters. If `None`, the
            model will use a uniform prior distribution, treating all parameters equally without any initial bias.
        :param algorithm_labels: Labels for the algorithms.
            This optional parameter allows to provide custom labels for the algorithms being compared. These labels
            will be used in the analysis and output of the model. If `None`, the algorithms will be labeled numerically.
        :param seed: Random seed for reproducibility. Default is 42.
            This parameter ensures that the random processes in the model are consistent across runs, allowing for
            reproducible results.
        """
        super(BayesianPlackettLuceTest, self).__init__('bayesian_plackett-luce_test.stan', None, seed=seed)

        # Ensure the input matrix does not contain NaN values
        assert not np.any(np.isnan(x_matrix)), ('The dataset x contains NaN values. '
                                                'Please remove or handle these NaNs before proceeding.')

        # Set the prior to ones if not provided
        if prior is None:
            prior = np.ones(x_matrix.shape[1])

        # Ensure the length of the prior matches the number of columns in the matrix
        assert len(prior) == x_matrix.shape[1], ('The length of the prior vector has to be equal to the number of '
                                                 'columns of x_matrix.')

        self.x_matrix = x_matrix
        self.prior = prior
        self.minimize = minimize
        self.algorithm_labels = algorithm_labels
        self.col_names = None

        # Save rating matrix for PPC check
        self.ranking_matrix = None

    def _transform_data(self) -> dict:
        """
        Transform the input data for the Stan model.

        :return: Dictionary containing transformed data.
        """
        # Modify the matrix based on whether we are minimizing or maximizing
        aux = self.x_matrix if self.minimize else -1 * self.x_matrix

        # Create a ranking matrix based on the transformed values
        ranking_matrix = np.apply_along_axis(lambda i: stats.rankdata(i, method='ordinal'), axis=1, arr=aux)

        # Set column names based on algorithm labels or generate default names
        if self.algorithm_labels is None:
            self.col_names = [f'V{i}' for i in range(1, self.x_matrix.shape[1] + 1)]
        else:
            self.col_names = self.algorithm_labels
        row_names = [f'R{i}' for i in range(1, self.x_matrix.shape[0] + 1)]

        # Create a DataFrame for the ranking matrix
        ranking_matrix_df = pd.DataFrame(ranking_matrix, columns=self.col_names, index=row_names)

        self.ranking_matrix = ranking_matrix
        return dict(
            n=ranking_matrix_df.shape[0],  # Number of instances
            m=ranking_matrix_df.shape[1],  # Number of algorithms
            ranks=ranking_matrix_df.values,  # Ranking matrix
            alpha=self.prior,  # Prior distribution for the model parameters
            weights=np.ones(ranking_matrix_df.shape[0])  # Weights for each instance
        )

    def _posterior_predictive_check(self, directory_path: str, file_path: str,
                                    file_name: str = 'posterior_predictive_check', font_size: int = 12,
                                    save: bool = True) -> None:
        """
        This function performs posterior predictive checks and generates plots comparing the observed data
        to the posterior predictive distributions.

        :param directory_path: Path to the parent directory where the files are stored.
        :param file_path: Path where the plot should be saved.
        :param file_name: Name of the file to save the plot. Default is 'posterior_predictive_check'.
        :param font_size: Font size for the plot text elements. Default is 12
        :param save: Whether to save the plot to file. Default is True.
        :return: None
        """
        print(f'{datetime.now().time().strftime("%H:%M:%S")} - INFO: Running posterior predictive check.')
        # Retrieve posterior predictive samples
        ranks = self.ranking_matrix  # shape: (n_instances, n_algorithms)
        ranks_rep = self._fit.stan_variable('ranks_rep')  # shape: (n_cd, n_instances, n_algorithms)
        n_cd, n_instances, n_algorithms = ranks_rep.shape
        n_draws = int(n_cd / self.chains)

        ranks_rep_dict = dict()
        observed_ranks_dict = dict()

        print('\nPosterior Predictive Check Metrics')
        # Create InferenceData object for each algorithm
        for alg in range(n_algorithms):
            observed_ranks_alg = ranks[:, alg]  # (n_instances)
            ranks_rep_alg = ranks_rep[:, :, alg]  # shape (n_cd, n_instances,)

            # Calculate and print PPC Metrics
            metrics = [posterior_predictive_check_metrics(observed_ranks_alg, ranks_rep_alg[i], ranks=True) for i in
                       range(n_cd)]

            means, std_devs = calculate_statistics(metrics)
            print(f'{self.algorithm_labels[alg]}\nMeans: {means}\nStdDevs: {std_devs}\n')

            ranks_rep_alg = ranks_rep_alg.reshape(
                (self.chains, n_draws, n_instances))  # shape (chains, n_draws, n_instances)
            ranks_rep_dict[f'ranks_rep_alg_{alg}'] = ranks_rep_alg
            observed_ranks_dict[f'ranks_alg_{alg}'] = observed_ranks_alg

        # Add groups to InferenceData
        self.inf_data.add_groups(posterior_predictive=ranks_rep_dict)

        # Suppress warning of incorrect dimensions for observed data
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="the default dims 'chain' and 'draw' will be added automatically")
            self.inf_data.add_groups(observed_data=observed_ranks_dict)

        # Generate the PPC plot
        variables = [self.algorithm_labels[alg] for alg in range(n_algorithms)]
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
                directory_path: str = 'results', file_path: str = 'bayesian_plackett_luce_test',
                file_name: Optional[str] = None, **kwargs) -> dict:
        """
        Analyse the results using the Bayesian Plackett-Luce model.

        :param posterior_predictive_check: Whether to do a posterior predictive check. Default is True.
        :param plot: Whether to generate and display plots. Default is True.
        :param save: Whether to save the results and plots to files. Default is True.
        :param round_to: Number of decimal places to round the results to. Default is 4.
        :param directory_path: Path to the parent directory where the files are stored. Default is 'results'.
        :param file_path: Directory path to save the files. Default is 'bayesian_plackett_luce_test'.
        :param file_name: Name of the file to save the results and plots. If None, a default name based on the current timestamp will be used.
        :param kwargs: Additional keyword arguments for customized analysis and plotting.
        :return: A dictionary containing the analysis results, including posterior probabilities and additional details.
        """
        # Perform PPC check
        if posterior_predictive_check:
            file_name_ppc = f'{self._execution_time}' if file_name is None else file_name
            self._posterior_predictive_check(directory_path, file_path, file_name=f'{file_name_ppc}_ppc', font_size=10,
                                             save=save)
        # Perform a simple analysis and print the summary
        self._simple_analysis()

        # Extract posterior ratings
        posterior = self.inf_data.posterior['ratings'].values

        # Reshape and calculate the mean of the posterior samples
        posterior = posterior.reshape((self.chains, self.iter_sampling, -1)).mean(axis=0)

        # Convert posterior samples to a DataFrame
        posterior_df = pd.DataFrame(posterior, columns=self.col_names)

        # Calculate the posterior mode for rankings
        posterior_mode = posterior_df.apply(lambda x: stats.rankdata(-x, method='min'), axis=1)

        # Calculate expected mode rank and win probability
        expected_mode_rank = posterior_mode.mean(axis=0)
        expected_win_prob = posterior_df.mean(axis=0)

        expected_mode_rank = pd.Series(expected_mode_rank, index=self.col_names)
        expected_win_prob = pd.Series(expected_win_prob, index=self.col_names)

        # Compile the parameters used in the analysis into a dictionary
        parameters = dict(
            prior=self.prior,  # Prior distribution used in the model
            iter_sampling=self.iter_sampling,  # Number of draws from the posterior for each chain
            iter_warmup=self.iter_warmup,  # Number of warm-up (burn-in) samples
            chains=self.chains,  # Number of chains in the MCMC sampling
            sampling_parameters=self.sampling_parameters,  # Additional parameters for the MCMC sampling
            seed=self.seed  # Random seed for reproducibility
        )

        additional = dict(
            posterior_mode=self.inf_data.posterior,  # Posterior samples from the model
            ranking_matrix=self.ranking_matrix,  # Original ranking matrix used in the analysis
        )
        # Compile the results into a dictionary
        results = dict(
            method='Bayesian Plackett-Luce model',  # Method used for the analysis
            inference_data=self.inf_data,  # arviz InferenceData: Container for inference data storage using xarray.
            parameters=parameters,  # Parameters used in the analysis
            posterior_weights=posterior_df,  # DataFrame of posterior weights
            expected_win_prob=expected_win_prob,  # DataSeries of expected win probabilities for each algorithm
            expected_mode_rank=expected_mode_rank,  # DataSeries of expected mode ranks for each algorithm
            additional=additional  # Additional details from the analysis
        )

        # Save results if requested
        if save:
            self.save_results(results, directory_path=directory_path, file_path=file_path, file_name=file_name)

        # Print the rounded results
        rounded_results = print_result(results, round_to=round_to)
        print(f'\nExpected Win Probability:\n{rounded_results["expected_win_prob"]}')
        print(f'\nExpected Mode Rank:\n{rounded_results["expected_mode_rank"]}')

        # Plot the posterior weights if requested
        if plot:
            weights = posterior_df
            plot_boxplot(weights, show_plt=False)

            # Save the plot if requested
            if save:
                self.save_plot(directory_path=directory_path, file_path=file_path, file_name=file_name)
                plt.show()
            else:
                plt.tight_layout()
                plt.show()
        return results
