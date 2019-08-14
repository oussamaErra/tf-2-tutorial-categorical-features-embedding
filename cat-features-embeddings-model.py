import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import shap
from sklearn.preprocessing import StandardScaler

data,labels = shap.datasets.adult(display=True)
data.select_dtypes('category').columns

def generate_categorical_feature_tf(categorical_features,num_features,data):
    models= []
    inputs = []
    for cat in categorical_features:
        vocab_size = data[cat].nunique()
        inpt = tf.keras.layers.Input(shape=(1,),name='input_'+'_'.join(cat.split(' ')))
        inputs.append(inpt)
        embed = tf.keras.layers.Embedding(vocab_size,200,\
                                          trainable=True,embeddings_initializer=tf.initializers.random_normal)(inpt)
        embed_rehsaped =tf.keras.layers.Reshape(target_shape=(200,))(embed)
        models.append(embed_rehsaped)
    num_input = tf.keras.layers.Input(shape=(len(num_features)),\
                                      name='input_number_features')
    inputs.append(num_input)
    models.append(num_input)
    merge_models= tf.keras.layers.concatenate(models)
    pre_preds = tf.keras.layers.Dense(1000)(merge_models)
    pre_preds = tf.keras.layers.BatchNormalization()(pre_preds)
    pre_preds = tf.keras.layers.Dense(1000)(pre_preds)
    pre_preds = tf.keras.layers.BatchNormalization()(pre_preds)
    pred = tf.keras.layers.Dense(1,activation='sigmoid')(pre_preds)
    model_full = tf.keras.models.Model(inputs= inputs,\
                                       outputs =pred)
    model_full.compile(loss=tf.keras.losses.binary_crossentropy,\
                       metrics=['accuracy'],
                       optimizer='adam')
    return model_full

def prepar_data_set(data_df):
    categoy_features = data_df.select_dtypes('category').columns
    numerique_features = data_df.select_dtypes('number').columns
    for col in categoy_features:
        encoder = LabelEncoder()
        data_df[col] = encoder.fit_transform(data_df[col])
    return data_df,categoy_features,numerique_features

train,cat_features,num_featture = prepar_data_set(data)

model = generate_categorical_feature_tf(cat_features,num_featture,train)



scaler = StandardScaler()
train[num_featture] = scaler.fit_transform(train[num_featture])
input_dict= {
    'input_Workclass':train[cat_features[0]],
    "input_Marital_Status":train[cat_features[1]],
    "input_Occupation":train[cat_features[2]],
    "input_Relationship":train[cat_features[3]],
    "input_Race":train[cat_features[4]],
    "input_Sex":train[cat_features[5]],
    "input_Country":train[cat_features[6]],
    "input_number_features": train[num_featture]
}

model.fit(input_dict,labels*1,epochs=50,batch_size=64,class_weight=\
    {0:0.5,1:0.5})


