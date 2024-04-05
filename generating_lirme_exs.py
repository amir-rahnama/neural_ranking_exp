import pickle 
from lime import lime_tabular
import shap
import pandas as pd
import re
import numpy as np
import tensorflow as tf
import time
from lirme import LIRME

tf.random.set_seed(42)
np.random.seed(42)


def predict_GAM(instances):
    """Predicts the documents' ranking scores using the ranking model

    Args:
        instances: pandas series of documents returned by a specific query
        idx: index (integer) of the document to explain in instances
    Returns:
        A list of scores (float) of equal length as instances
    """
    instances = serialize_all(instances)
    tf_example_predictor = _loaded_model.signatures['predict']
    scores = tf_example_predictor(tf.convert_to_tensor(instances))['output']
    return scores.numpy().flatten()


def _float_feature(value):
    """Converts a numerical value into a TensorFlow Feature object of type FloatList"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Converts a numerical value into a TensorFlow Feature object of type Int64List"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def transform_lime_exp(exp, features):
    transform_exp = np.zeros(len(features))

    for i in range(len(exp)):
        feature_idx = np.array(find_feature_idx(exp[i][0]))
        transform_exp[feature_idx] = exp[i][1]
    
    return transform_exp

def find_feature_idx(feature_rule):
    res = re.findall(r'feature_id_(\d+)', feature_rule)
    if len(res) > 0:
        return int(res[0])


def serialize_all(examples):
    """Converts a dataframe of documents into a list of TensorFlow Example"""
    list_examples = []
    for idx, row in examples.iterrows():
        example_dict = {
            f'{feat_name}': _float_feature(feat_val) for
            feat_name, feat_val in zip(_name_features, row.iloc[2:].tolist())
        }

        example_dict['relevance_label'] = _int64_feature(int(row['relevance_label']))

        example_proto = tf.train.Example(features=tf.train.Features(feature=example_dict))
        list_examples.append(example_proto.SerializeToString())

    return list_examples
    
def transform_lime_exp(exp, features):
    transform_exp = np.zeros(len(features))

    for i in range(len(exp)):
        feature_idx = np.array(find_feature_idx(exp[i][0]))
        transform_exp[feature_idx] = exp[i][1]
    
    return transform_exp


def shap_exp(instances, train_data, pred_fn, sample_size):
    #sample_size_background = 100
    #s = shap.sample(train_data, sample_size_background)
    explainer = shap.KernelExplainer(pred_fn, train_data)
    shap_exps = explainer.shap_values(instances, nsamples=sample_size)
    
    return shap_exps

def random_exp(size):
    return np.random.dirichlet(np.ones(size), size=1).flatten()


def exp_pred_fn(instance):
    _name_features = [str(i + 1) for i in range(0, 100)]
    new_sample = pd.DataFrame(instance, columns=_name_features)
    new_sample.insert(0, 'relevance_label', np.repeat(0, instance.shape[0]))
    
    return predict_GAM(new_sample)
    

def lime_exp(instance, pred_fn, train_data, sample_size):
    feature_names = ['feature_id_' + str(i) for i in np.arange(train_data.shape[1])]
    lime_exp = lime_tabular.LimeTabularExplainer(train_data, 
                                             kernel_width=3, verbose=False, mode='regression',
                                             feature_names=feature_names)

    exp = lime_exp.explain_instance(instance, pred_fn, num_features=train_data.shape[1],  
                                    num_samples=sample_size)
    lime_e = exp.as_list()
    lime_e_trans = transform_lime_exp(lime_e, feature_names)
    
    return lime_e_trans



if __name__ == '__main__':
    p_exps = {}
    pointwise_exps = ['lirme', 'exs_top_k', 'exs_score', 'exs_regression', 'lime', 'shap', 'random']
    qids = [28587, 23068, 28700, 24240, 23890, 28483, 23132, 29379, 23022, 24923, 25000, 29889, 25191, 25207, 25226,
                28035, 28203, 28670, 28286, 28276]
    
    df_test = pd.read_csv('./test_yahoo.csv')
    grouped_qid = df_test.groupby('qid')
    _name_features = [str(i + 1) for i in range(0, 100)]
    
    start = time.time()
    _model_yahoo = './model'
    # Check if GPU is available
    if tf.config.list_physical_devices('GPU'):
        # If GPU is available, set the appropriate strategy
        strategy = tf.distribute.MirroredStrategy()  # Automatically use all available GPUs
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
        # Define and compile your model inside the strategy's scope
        with strategy.scope():
            # Load the model
            _loaded_model = tf.saved_model.load(_model_yahoo)
    else:
        # If no GPU is available, load the model without any special strategy
        _loaded_model = tf.saved_model.load(_model_yahoo)
        
    end = time.time()
    print('Model loaded in ', (end - start)/60)

    background = df_test.iloc[:1000, 2:].values
    print('background shape', background.shape)
    lirme = LIRME(background)
    ranked_pos = 2
    
    for e_sample_size in [500, 1000, 2000, 3000, 4000, 5000]: 
    #for e_sample_size in [500]: 
        for key in qids:
            print('Document explanations for query', key)
            p_exps[key] = {}
            
            for t in pointwise_exps:
                p_exps[key][t] = {}
            
            grouped_data = grouped_qid.get_group(key)
            pred_q_docs = predict_GAM(grouped_data)
            
            base_rank = np.argsort(pred_q_docs)[::-1]
            explained_instance_idx = np.argwhere(base_rank == ranked_pos)[0][0]
        
            instance =  grouped_data.iloc[explained_instance_idx, 2:]

            print('instance shape', instance.shape)
           
            '''l_exp = lime_exp(instance, exp_pred_fn, background, sample_size=e_sample_size)
            p_exps[key]['lime'] = l_exp
            
            s_exp = shap_exp(instance, background, exp_pred_fn, sample_size=e_sample_size)
            p_exps[key]['shap'] = s_exp'''
            
            #instance =  grouped_data.iloc[explained_instance_idx, 1:]
            
            '''exp_lirme = lirme.explain(instance, exp_pred_fn, pred_q_docs, 
                        sur_type='ridge', label_type='regression', 
                        instance_idx=explained_instance_idx, top_rank=ranked_pos, sample_size=e_sample_size)
            p_exps[key]['lirme'] = exp_lirme'''
        
            exp_exs_v1 = lirme.explain(instance, exp_pred_fn, pred_q_docs, 
                        sur_type='svm', label_type='top_k_binary', 
                        instance_idx=explained_instance_idx, top_rank=ranked_pos, sample_size=e_sample_size)
            p_exps[key]['exs_top_k'] = exp_exs_v1
        
            '''exp_exs_v2 = lirme.explain(instance, exp_pred_fn, pred_q_docs, 
                        sur_type='svm', label_type='score', 
                        instance_idx=explained_instance_idx, top_rank=ranked_pos, sample_size=e_sample_size)
            p_exps[key]['exs_score'] = exp_exs_v2'''
        
            exp_exs_v3 = lirme.explain(instance, exp_pred_fn, pred_q_docs, 
                        sur_type='svm', label_type='top_k_rank', 
                        instance_idx=explained_instance_idx, top_rank=ranked_pos, sample_size=e_sample_size)
            p_exps[key]['top_k_rank'] = exp_exs_v3
        
            '''p_exps[key]['random'] = random_exp(instance.shape[0] - 1)'''
        
        pickle.dump(p_exps, open( "./p_exps_{}_v2_ranked_{}_anchored_{}.p".format(e_sample_size, ranked_pos, ranked_pos), "wb" ) )
