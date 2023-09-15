import pandas as pd
import os
import json
from chatgpt_validation import ask_for_repo
from tabulate import tabulate


nodes_dict_swapped = {
    "PCiC": "Compound",
    "_PCiC": "Pharmacological class",
    "GcG": "Gene",
    "_GcG": "Gene",
    "GiG": "Gene",
    "Gr>G": "Gene",
    "_Gr>G": "Gene",
    "GpBP": "Biological Process",
    "_GpBP": "Gene",
    "GpCC": "Cellular component",
    "_GpCC": "Gene",
    "GpMF": "Molecular function",
    "_GpMF": "Gene",
    "GpPW": "Pathway",
    "_GpPW": "Gene",
    "AdG": "Gene",
    "_AdG": "Anatomy",
    "AeG": "Gene",
    "_AeG": "Anatomy",
    "AuG": "Gene",
    "_AuG": "Anatomy",
    "DaG": "Gene",
    "_DaG": "Disease",
    "DdG": "Gene",
    "DpS": "Symptom",
    "_DpS": "Disease",
    "_DdG": "Disease",
    "DuG": "Gene",
    "_DuG": "Disease",
    "DlA": "Anatomy",
    "_DlA": "Disease",
    "DrD": "Disease",
    "_DrD": "Disease",
    "CbG": "Gene",
    "_CbG": "Compound",
    "CdG": "Gene",
    "_CdG": "Compound",
    "CcSE": "Side effect",
    "_CcSE": "Compound",
    "CpD": "Disease",
    "CrC": "Compound",
    "_CrC": "Compound",
    "_CpD": "Compound",
    "CtD": "Disease",
    "_CtD": "Compound",
    "CuG": "Gene",
    "_CuG": "Compound"
}


nodes_dict_swapped['_GiG'] = 'Gene'
nodes_dict_swapped['_Gr>G'] = 'Gene'
nodes_dict_swapped['Gr>G'] = 'Gene'

nodes_dict_final_node = {
    "PCiC": "Compound",
    "_PCiC": "Pharmacological class",
    "GcG": "Gene",
    "_GcG": "Gene",
    "GiG": "Gene",
    "Gr>G": "Gene",
    "_Gr>G": "Gene",
    "GpBP": "Biological Process",
    "_GpBP": "Gene",
    "GpCC": "Cellular component",
    "_GpCC": "Gene",
    "GpMF": "Molecular function",
    "_GpMF": "Gene",
    "GpPW": "Pathway",
    "_GpPW": "Gene",
    "AdG": "Gene",
    "_AdG": "Anatomy",
    "AeG": "Gene",
    "_AeG": "Anatomy",
    "AuG": "Gene",
    "_AuG": "Anatomy",
    "DaG": "Gene",
    "_DaG": "Disease",
    "DdG": "Gene",
    "DpS": "Symptom",
    "_DpS": "Disease",
    "_DdG": "Disease",
    "DuG": "Gene",
    "_DuG": "Disease",
    "DlA": "Anatomy",
    "_DlA": "Disease",
    "DrD": "Disease",
    "_DrD": "Disease",
    "CbG": "Gene",
    "_CbG": "Compound",
    "CdG": "Gene",
    "_CdG": "Compound",
    "CcSE": "Side effect",
    "_CcSE": "Compound",
    "CpD": "Disease",
    "CrC": "Compound",
    "_CrC": "Compound",
    "_CpD": "Compound",
    "CtD": "Disease",
    "_CtD": "Compound",
    "CuG": "Gene",
    "_CuG": "Compound"
}

nodes_dict_initial_node = {
    "_PCiC": "Compound",
    "PCiC": "Pharmacological class",
    "_GcG": "Gene",
    "GcG": "Gene",
    "GiG": "Gene",
    "_Gr>G": "Gene",
    "Gr>G": "Gene",
    "_GpBP": "Biological Process",
    "GpBP": "Gene",
    "_GpCC": "Cellular component",
    "GpCC": "Gene",
    "_GpMF": "Molecular function",
    "GpMF": "Gene",
    "_GpPW": "Pathway",
    "GpPW": "Gene",
    "_AdG": "Gene",
    "AdG": "Anatomy",
    "_AeG": "Gene",
    "AeG": "Anatomy",
    "_AuG": "Gene",
    "AuG": "Anatomy",
    "_DaG": "Gene",
    "DaG": "Disease",
    "_DdG": "Gene",
    "_DpS": "Symptom",
    "DpS": "Disease",
    "DdG": "Disease",
    "_DuG": "Gene",
    "DuG": "Disease",
    "_DlA": "Anatomy",
    "DlA": "Disease",
    "DrD": "Disease",
    "_DrD": "Disease",
    "_CbG": "Gene",
    "CbG": "Compound",
    "_CdG": "Gene",
    "CdG": "Compound",
    "_CcSE": "Side effect",
    "CcSE": "Compound",
    "_CpD": "Disease",
    "CrC": "Compound",
    "_CrC": "Compound",
    "CpD": "Compound",
    "_CtD": "Disease",
    "CtD": "Compound",
    "_CuG": "Gene",
    "CuG": "Compound"
}


nodes_dict_initial_node['_GiG'] = 'Gene'
nodes_dict_initial_node['_Gr>G'] = 'Gene'
nodes_dict_initial_node['Gr>G'] = 'Gene'

neo4j_relation = {
    "PCiC": "INTERACTS_PCiC",
    "GcG": "COVARIES_GcG",
    "GiG": "INTERACTS_GiG",
    "Gr>G": "REGULATES_GrG",
    "GpBP": "PARTICIPATES_GpBP",
    "GpCC": "PARTICIPATES_GpCC",
    "GpMF": "PARTICIPATES_GpMF",
    "GpPW": "PARTICIPATES_GpPW",
    "AdG": "DOWNREGULATES_AdG",
    "AeG": "EXPRESSES_AeG",
    "AuG": "UPREGULATES_AuG",
    "DaG": "ASSOCIATES_DaG",
    "DpS": "PRESENTS_DpS",
    "DdG": "DOWNREGULATES_DdG",
    "DuG": "UPREGULATES_DuG",
    "DlA": "LOCALIZES_DlA",
    "DrD": "RESEMBLE_DrD",
    "CbG": "BINDS_CbG",
    "CdG": "DOWNREGULATES_CdG",
    "CcSE": "CAUSES_CcSE",
    "CrC": "RESEMBLES_CrC",
    "CpD": "PALIATES_CpD",
    "CtD": "TREATS_CtD",
    "CuG": "UPREGULATES_CuG"
}

rel_dict = { "Pharmacologic Class includes compound": "PCiC",
    "Compound includes Pharmacologic Class": "_PCiC",
    "Gene covaries Gene": "GcG",
    "Gene is covaried by Gene": "_GcG",
    "Gene interacts Gene": "GiG",
    "Gene is interacted by Gene": "_GcG",
    "Gene regulates Gene": "Gr>G",
    "Gene regulates Gene": "_Gr>G",
    "Gene is regulated by Gene": "_GcG",
    "Gene participates in Biological Process": "GpBP",
    "Biological process includes gene": "_GpBP",
    "Gene participates in Cellular component": "GpCC",
    "Cellular component includes gene": "_GpCC",
    "Gene participates in Molecular function": "GpMF",
    "Molecular function includes gene": "_GpMF",
    "Gene participates in Pathway": "GpPW",
    "Pathway includes gene": "_GpPW",
    "Anatomy downregulates gene": "AdG",
    "Gene is downregulated by anatomy": "_AdG",
    "Anatomy expresses gene": "AeG",
    "Gene is expressed by anatomy": "_AeG",
    "Anatomy upregulates gene": "AuG",
    "Gene is upregulated by anatomy": "_AuG",
    "Disease associates gene": "DaG",
    "Gene is associated to diseasse": "_DaG",
    "Disease downregulates gene": "DdG",
    "Disease presents symptom": "DpS",
    "Symptom is presented by diseasse": "_DpS",
    "Gene is downregulated to diseasse": "_DdG",
    "Disease upregulates gene": "DuG",
    "Gene is upregulated to diseasse": "_DuG",
    "Disease localizes anatomy": "DlA",
    "Anatomy is localized to diseasse": "_DlA",
    "Disease resembles Disease": "DrD",
    "Compound binds gene": "CbG",
    "Gene is binded by compound": "_CbG",
    "Compound downregulates gene": "CdG",
    "Gene is downregulated by compound": "_CdG",
    "Compound causes side effect": "CcSE",
    "Side effect is caused by compound": "_CcSE",
    "Compound palliates disease": "CpD",
    "Disease is palliated by compound": "_CpD",
    "Compound resembles compound": "CrC",
    "Disease is palliated by compound": "_CpD",
    "Compound treats disease": "CtD",
    "Disease is treated by compound": "_CtD",
    "Compound upregulates gene": "CuG",
    "Gene is upregulated by compound": "_CuG"}

rel_dict = {y: x for x, y in rel_dict.items()}
rel_dict['_GiG'] = 'Gene interacts Gene'
rel_dict['_Gr>G'] = 'Gene regulates Gene'
rel_dict['Gr>G'] = 'Gene regulates Gene'


rel_name_dict = {
    "PCiC": "Pharmacologic Class includes compound",
    "_PCiC": " includes ",
    "GcG": " covaries ",
    "_GcG": " is covaried by ",
    "GiG": " interacts ",
    "_GiG": " is interacted by ",
    "Gr>G": " regulates ",
    "_Gr>G": " is regulated by ",
    "GpBP": " participates in ",
    "_GpBP": " includes ",
    "GpCC": " participates in ",
    "_GpCC": " includes ",
    "GpMF": " participates in ",
    "_GpMF": " includes ",
    "GpPW": " participates in ",
    "_GpPW": " includes ",
    "AdG": " downregulates ",
    "_AdG": " is downregulated by ",
    "AeG": " expresses ",
    "_AeG": " is expressed by ",
    "AuG": " upregulates ",
    "_AuG": " is upregulated by ",
    "DaG": " associates ",
    "_DaG": " is associated to ",
    "DdG": " downregulates ",
    "DpS": " presents ",
    "_DpS": " is presented by ",
    "_DdG": " is downregulated to ",
    "DuG": " upregulates ",
    "_DuG": " is upregulated to ",
    "DlA": " localizes ",
    "_DlA": " is localized to ",
    "DrD": " resembles ",
    "_DrD": " resembles ",
    "CbG": " binds ",
    "_CbG": " is binded by ",
    "CdG": " downregulates ",
    "_CdG": " is downregulated by ",
    "CcSE": " causes ",
    "_CcSE": " is caused by ",
    "CpD": " palliates ",
    "_CpD": " is palliated ",
    "_CrC": " resembles ",
    "_CpD": " is palliated ",
    "CtD": " treats ",
    "_CtD": " is treated by ",
    "CuG": " upregulates ",
    "_CuG": " is upregulated by ",
    "CrC": " resembles ",
    "_CrC": " resembles ",
    "Gr>G": " regulates ",
    "_Gr>G": " regulates ",
    "_GiG": " interacts ",
}

def rule2read(metapath, score = 0):
    nodes = []
    relations = []
    for step in metapath:
        nodes.append(nodes_dict_initial_node[step])
        relations.append(rel_name_dict[step])

    nodes.append(nodes_dict_final_node[metapath[-1]])
    latex_string = "["

    for i, node in enumerate(nodes[:-1]):
        latex_string += f"{node} $\\stackrel{{\\text{{ {relations[i]} }} }}{{\\longrightarrow}}$ "

    latex_string += f"{nodes[-1]}"

    latex_string += "]"
    # latex_string += f"] score = {score}"

    print(latex_string)

    return latex_string


def create_names_df_diseases(entities, diseases):
    entities[['Type', 'doid']] = entities['entity'].str.split('::', 1, expand=True)
    entities = entities[entities['Type']=='Disease']
    entities['doid'] = entities['doid'].str.replace(':','_')
    entities['name']= entities['doid'].map(diseases)

    save = False
    if save:
        with open('diseases.txt', 'w') as fd:
            entities[['id','entity','doid','name']].to_csv(fd, header=None, index=None, sep='\t', mode='w',line_terminator='\n')
    return entities[['id','entity','doid','name']]


def create_names_df_drug(entities, id2name_drug,name):
    entities[['Type', 'drugbank_id']] = entities['entity'].str.split('::', 1, expand=True)
    entities = entities[entities['Type']=='Compound']
    entities['drugbank_id'] = entities['drugbank_id'].str.replace(':','_')
    entities['name']= entities['drugbank_id'].map(id2name_drug)
    entities['name'] = entities['name'].str.replace('\r','')

    save = False
    if save:
        with open('drugs.txt', 'w') as fd:
            entities[['id','entity','drugbank_id','name']].to_csv(fd, header=None, index=None, sep='\t', mode='w',line_terminator='\n')
    return entities[['id','entity','drugbank_id','name']]


def get_query_info(tops,rule_folder,file_save_readable,diseases_df,save_test_queries_file):

    # This function writes the head, the relation, the tail and the rank in a file
    # Look for correct tail and rank
    head = list(tops.keys())[0].split('->')[0]
    head_name = id2name_drug[head.split('::')[-1]][:-1]

    id_head = entities['id'][entities['entity'] == head]
    true_tail = ranks['tail_id'][ranks['head_id'] == int(id_head)]
    for i in range(len(true_tail)):
        true_tail_i = list(true_tail)[i]
        true_tail_compound = entities['entity'][entities['id'] == true_tail_i]
        true_tail_name = diseases[str(list(true_tail_compound)[0]).split('::')[-1].replace(':', '_')]

        # Arreglar para que haya queries con heads distintas
        rankL = ranks['lrank'][ranks['head_id'] == int(id_head)]
        rankH = ranks['hrank'][ranks['head_id'] == int(id_head)]

        # Degree info
        degree_info = float(diseases_df['InvD'][diseases_df['doid'] == true_tail_compound.iloc[0].split('::')[-1].replace(':', '_')])

        print(true_tail_name + ' which is treated by ' + head_name + ' appears in rank ' + str(
            list(rankL)[0]) + '-' + str(list(rankH)[0]))
        query_rank = [true_tail_name, ' which is treated by ', head_name, ' appears in rank ', str(list(rankL)[0]), '-',
                      str(list(rankH)[0])]

        query_summary = [head_name,true_tail_name,str(list(rankL)[0]),str(list(rankH)[0]),degree_info]

        with open( save_test_queries_file, 'a') as fd:
            pd.DataFrame(query_summary).T.to_csv(fd, header=None, index=None, sep=' ', mode='w', line_terminator='\n')


        with open(rule_folder + file_save_readable, 'a') as fd:
            pd.DataFrame(query_rank).T.to_csv(fd, header=None, index=None, sep=' ', mode='w', line_terminator='\n')


def insert_substring(original_string, substring, position):
    return original_string[:position] + substring + original_string[position:]


def generate_cypher_query(initial_node_label, final_node_label,metapath,
                          initial_node_name=None, final_node_name = None):
    """
    Generates a Cypher query string for obtaining a path that follows the given metapath between
    an initial and final node.

    :param initial_node_label: The label of the initial node.
    :param final_node_label: The label of the final node.
    :param metapath: A list of relationship types in the metapath.
    :return: A Cypher query string.
    """

    if initial_node_name is not None:
        query = f"MATCH path = ({initial_node_label}"+ '{name:' + f"'{initial_node_name}'" + '})'
    else:
        query = f"MATCH path = ({initial_node_label})"
    for rel in metapath:


        if rel.startswith("_"):
            # The relationship is inverse, so we need to reverse the arrow direction
            node = nodes_dict_swapped[rel]

            # query += f"<-[:{neo4j_relation[rel[1:]]}]- ({node})"
            query += f"<-[:{neo4j_relation[rel[1:]]}]- ()"
        else:
            node = nodes_dict_swapped[rel]
            # The relationship is not inverse, so we can use the default arrow direction
            # query += f"-[:{neo4j_relation[rel]}]-> ({node})"
            query += f"-[:{neo4j_relation[rel]}]-> ()"
    # query += f"({final_node_label}) RETURN path"
    if final_node_name is not None:
        query = query[:-1] + '{name:' + f"'{final_node_name}'" + '})'

    query += ' RETURN path'
    print(query)
    return query


def manual_formatting(table):
    # Generate LaTeX table in booktabs format
    table_str = r"\begin{tabular}{ll}" + "\n"
    table_str += r"\toprule" + "\n"
    table_str += r"Score & Rule \\" + "\n"
    table_str += r"\midrule" + "\n"

    for index, row in table.iterrows():
        name = str(row['Score'])
        age = str(row['Rule'])
        table_str += name + " & " + str(age) + r" \\" + "\n"

    table_str += r"\bottomrule" + "\n"
    table_str += r"\end{tabular}"

    print(table_str)
    return table_str

if __name__ == "__main__":

    # Define parameters for API request
    model_engine = "text-davinci-003" # Specify the GPT model to use
    temperature = 0.3 # Controls the creativity of the responses
    max_tokens = 100 # Controls the length of the response


    # File names
    dataset_dir = '/home/ana/Proyectos/rnnlogic_2/RNNLogic+/data/20230904_1800k_mj_rules/'
    id2drug = '/home/ana/Proyectos/got_test_tfm1/test_save_explainable_results_hetionet_rules_chatgpt' \
              '/drugbank id2name.csv'
    id2disease = '/home/ana/Proyectos/got_test_tfm1/test_save_explainable_results_hetionet_rules_chatgpt/do2name.json'

    """
        Write rule file
    """
    rule_folder = '/home/ana/Proyectos/got_test_tfm1/hetionet_final_results/metapathPLUS_22/CtD/'

    save_prediction_file = rule_folder[:-4]+'predictions.txt'
    save_test_queries_file = rule_folder[:-4] + 'test_queries.txt'
    entity_file = dataset_dir + 'entities.dict'

    """
        Write ranks file
    """
    ranks_file = '/home/ana/Proyectos/got_test_tfm1/hetionet_final_results/metapathPLUS_22/20230904_hetionet_mj_rules_PLUS_22_100_100_ranks_1.csv'


    # Diseases
    f = open(id2disease)
    diseases = json.load(f)


    # Diseases all ids
    diseases_df = pd.read_csv('/home/ana/Proyectos/got_test_tfm1/'
                              'test_save_explainable_results_hetionet_rules_chatgpt/diseases.txt',
                              names = ['id', 'entity', 'doid', 'name','InvD'], sep='\t',lineterminator='\n')

    diseases_df['name'] = diseases_df['name'].replace('\r','')
    # Drugs
    drugs = pd.read_csv(id2drug, names = ['id', 'name'],sep='\t', lineterminator='\n')
    id2name_drug = drugs.set_index('id').T.to_dict('records')[0]
    files = os.listdir(rule_folder)


    # Ranks
    ranks = pd.read_csv(ranks_file, names=['eval', 'head_id', 'relation', 'tail_id', 'lrank', 'hrank'],
                        sep=';', lineterminator='\n')
    ranks = ranks[ranks['eval']=='test']

    # Entities
    entities = pd.read_csv(entity_file,names=['id','entity'],sep='\t',lineterminator='\n')



    # Chatgpt answers
    chatgpt_file = '/home/ana/Proyectos/got_test_tfm1/Performance_code/chatgpt_answers.txt'
    chatgpt_df = pd.read_csv(chatgpt_file, names=['drug', 'disease', 'answer', 'q'],sep='\t',lineterminator='\n')
    for file in files:  # For each query
        file_save_readable = file[:-4] + '_read.txt'
        f = open(rule_folder + file)
        tops = json.loads(f.read())

        # Get the info of the query
        get_query_info(tops, rule_folder, file_save_readable,diseases_df,save_test_queries_file)

        latex_table_dict = dict()
        if True:
            for prediction in tops.keys():  # For each prediction
                latex_table_list = []

                predition_elements = prediction.split('->')
                head = predition_elements[0].split('::')[-1]  # Head


                head_name = id2name_drug[head][:-1]

                tail_hat_name = diseases[predition_elements[-1].split('::')[-1].replace(':','_')]
                rules = tops[prediction]
                print(head_name+' treats '+tail_hat_name)
                degree_info = float(diseases_df['InvD'][diseases_df['doid']==predition_elements[-1].split('::')[-1].replace(':','_')])


                # Aquí llamar a ChatGPT para headname y tail_hat_name

                # No tengo licencia  :( así que aquí llamar
                #chatGPT_answers = ask_for_repo(head_name,tail_hat_name, model_engine, temperature, max_tokens)


                # Buscar la predicción en el fichero y si no está poner un cero
                questions = chatgpt_df.loc[(chatgpt_df['drug'] == head_name) & (chatgpt_df['disease'] == tail_hat_name)]
                pred = [head_name, ' treats ', tail_hat_name, str(degree_info)]
                if len(questions) > 0:
                    chatgpt_answers = list(questions['answer'])
                    pred = pred + chatgpt_answers
                pred_df = pd.DataFrame(pred).T
                with open(save_prediction_file, 'a', encoding="utf-8") as f:
                    pred_df.to_csv(f, header=None, index=None, sep='\t', mode='w', lineterminator='\n')

                with open(rule_folder + file_save_readable, 'a', encoding="utf-8") as f:
                    pred_df.to_csv(f, header=None, index=None, sep='\t', mode='w', lineterminator='\n')
                for rule in rules:
                    if rule[-1] != 0:
                        rule_read = list(map(rel_dict.get, rule[1:-1]))
                        latex_str = rule2read(rule[1:-1], rule[-1])
                        latex_table_list.append([rule[-1], latex_str])
                        cypher_query = generate_cypher_query('Compound','',rule[1:-1], initial_node_name=head_name,
                                              final_node_name=tail_hat_name)
                        print(rule_read)
                        with open(rule_folder + file_save_readable, 'a') as fd:
                            pd.DataFrame(rule_read +[rule[-1], latex_str, cypher_query]).T.to_csv(fd, header=None, index=None, sep='\n', mode='w', lineterminator='\n')

                table_df = pd.DataFrame(latex_table_list, columns=['Score', 'Rule'])
                table = manual_formatting(table_df)
                # table = tabulate(table_df, headers='keys', tablefmt='latex', disable_numparse=True)
                latex_table ='\n'+ head_name+' treats ' +tail_hat_name+ '\n\\begin{table}[H]\n\\centering\n\\caption{Table Caption}\n\\resizebox{\\textwidth}{!}{\n' + table + '}\n\\label{tab:my_table}\n\\end{table}'


                with open('my_test4.txt', 'a') as file:
                    # Write the string to the file
                    file.write(latex_table)
               # If rules empty remove the candidate tail


                # Merge with ranks to know the rank of the correct answer. Check if the correct is in the top 10

