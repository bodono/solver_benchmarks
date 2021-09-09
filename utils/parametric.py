import os
import pandas as pd


def print_results_parametric(problem, dimension):
    """
    Print parametric problem results
    """
    print('[%s]' % problem)
    no_ws_file = os.path.join('.', 'results', 'parametric_problems',
                              'no warmstart',
                              problem,
                              'n%i.csv' % dimension
                              )

    no_ws_df = pd.read_csv(no_ws_file)

    # Store results
    results_file = os.path.join(".", "results", "parametric_problems",
                                "%s_results.txt" % problem.lower())
    print("Saving statistics to file %s" % results_file)
    f = open(results_file, "w")
    f.write('  (no warm start): \n')
    f.write('   - median time: %.4e sec\n' % no_ws_df['run_time'].median())
    f.write('   - mean time:   %.4e sec\n' % no_ws_df['run_time'].mean())
    f.write('   - median iter: %d\n' % no_ws_df['iter'].median())
    f.write('   - mean iter:   %d\n' % no_ws_df['iter'].mean())
    print("")

def compute_results_parametric(problems, dimensions):

    row_list = []
    statistics_file = os.path.join('.', 'results', 'parametric_problems',
                                   "statistics.csv")
    for p in problems:

        statistics_prob_file = os.path.join('.', 'results', 'parametric_problems',
                                            "statistics_%s.csv" % p.lower())
        row_list_prob = []
        for d in dimensions[p]:
            no_ws_file = os.path.join('.', 'results', 'parametric_problems',
                                      'no warmstart',
                                      p,
                                      'n%i.csv' % d
                                      )
            no_ws_df = pd.read_csv(no_ws_file)
            dict_stats = {
                    'problem': p,
                    'dimension': d,
                    'nows_mean_time': no_ws_df['run_time'].mean(),
                    'nows_median_time': no_ws_df['run_time'].median(),
                    'nows_mean_iter': no_ws_df['iter'].mean(),
                    'nows_median_iter': no_ws_df['iter'].median(),
                    }
            row_list_prob.append(dict_stats)

        # Store prob statistics
        prob_statistics = pd.DataFrame(row_list_prob)
        prob_statistics.to_csv(statistics_prob_file)
        row_list += row_list_prob


    # Create statistics dict
    statistics = pd.DataFrame(row_list)
    statistics.to_csv(statistics_file)
