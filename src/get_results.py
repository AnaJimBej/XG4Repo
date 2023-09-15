import os
import present_results_definitions



def main():
    # Test anyburl rules
    dataset_dir = '/home/ana/Proyectos/rnnlogic_2/RNNLogic+/data/18k/20230207_hetionet_18k/'
    folder = '/home/ana/Proyectos/got_test_tfm1/metrics_results/'
    files = os.listdir(folder)
    files_trials = set([i.split('.', 1)[0][0:-1] for i in files])
    n_test = 85


    # Test

    file = '/home/ana/Proyectos/got_test_tfm1/metrics_results/EM_1_all_direct_ranks_1.csv'
    # df_test = present_results_definitions.read_rnnlogic(file)
    # df_relation = present_results_definitions.get_train(dataset_dir+'train.txt')
    # # present_results_definitions.plot_relation_result(df_test, 'title', df_relation, [])

    # Experiment 0: compare results
    present_results_definitions.compare_models_RNNLogic(folder, files, 'EM', 'title', ntest=n_test)
    present_results_definitions.compare_models_RNNLogic(folder, files, 'PLUS', 'title',ntest=n_test)

    # Experiment 1
    for file in files:
        present_results_definitions.plot_evolution_RNNLogic(folder+file, 'title')

    # # Experiment 2: 100-100-4
    # file = 'C:/Users/anaji/OneDrive - Universidad Politécnica de Madrid/MUIT_OneDrive/TFM_GAPS/20230210_hetionet_metrics_results/metrics_results/metrics_result_100_100_4_1.csv'
    # present_results_definitions.plot_evolution_RNNLogic(file,'N:100, K:100, L=4 :')
    #
    # # Experiment 3: 100-100-4
    # file = 'C:/Users/anaji/OneDrive - Universidad Politécnica de Madrid/MUIT_OneDrive/TFM_GAPS/20230210_hetionet_metrics_results/metrics_results/metrics_result_100_100_4_1.csv'
    # present_results_definitions.plot_evolution_RNNLogic(file,'N:100, K:100, L=4 :')
    #
    #
    # # Experiment 4: RNNLogic miner
    # file = 'C:/Users/anaji/OneDrive - Universidad Politécnica de Madrid/MUIT_OneDrive/TFM_GAPS/20230210_hetionet_metrics_results/metrics_results/metrics_result_RnnlogicMiner_1.csv'
    # present_results_definitions.plot_evolution_RNNLogic(file,'RNNLogic miner - N:15, K:15, L=4 :')
    #
    # # Experiment 5: All direct priors
    # file = 'C:/Users/anaji/OneDrive - Universidad Politécnica de Madrid/MUIT_OneDrive/TFM_GAPS/20230210_hetionet_metrics_results/metrics_results/metrics_result_allDirect_100_100_4_1.csv'
    # present_results_definitions.plot_evolution_RNNLogic(file,'All direct - N:100, K:100, L=4 :')
    #
    #
    # # Experiment 6: plot results when the heads are filtered
    # file = 'C:/Users/anaji/OneDrive - Universidad Politécnica de Madrid/MUIT_OneDrive/TFM_GAPS/20230210_hetionet_metrics_results/metrics_results_EM_all_direct_ranks_1.csv'
    # present_results_definitions.plot_evolution_from_filtered_file(file,'N:100, K:100, L=4 :','EM')
    #
    # # Experiment 7: plot results per relation from ranking file
    # file = 'C:/Users/anaji/OneDrive - Universidad Politécnica de Madrid/MUIT_OneDrive/TFM_GAPS/20230210_hetionet_metrics_results/hetionet_18k_sample_test_anyburl_all_direct_EM_1_20_20_ranks_1.csv'
    # df_test = present_results_definitions.read_rnnlogic(file)
    # df_relation = present_results_definitions.get_train(dataset_dir+'train.txt')
    # present_results_definitions.plot_relation_result(df_test, 'title', df_relation, [])
    #
    #
    # # Filter test
    # filter_heads_in_test = False
    # if filter_heads_in_test:
    #     last_iter_rank = 'EM_5_all_direct_ranks_1.csv'
    #     rank_folder = 'C:/Users/anaji/OneDrive - Universidad Politécnica de Madrid/MUIT_OneDrive/TFM_GAPS/20230210_hetionet_metrics_results/'
    #     present_results_definitions.filter_test_by_heads(dataset_dir, rank_folder, last_iter_rank)



if __name__ == "__main__":
    main()
