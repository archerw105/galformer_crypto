import math
from math import floor
import numpy
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print("Using CPU")

# Helper functions (replacing transformer_helper_dc imports)
def positional_encoding(max_position, d_model):
    """Generate positional encoding matrix"""
    position = torch.arange(max_position).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        -(math.log(10000.0) / d_model))
    
    pe = torch.zeros(1, max_position, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    
    return pe

def create_look_ahead_mask(size1, size2):
    """Create look ahead mask for decoder"""
    mask = torch.triu(torch.ones(size1, size2), diagonal=1)
    return mask == 1

# Helper functions (replacing rolling_and_plot_dc imports)
def data_plot(data, title="Time Series Data", figsize=(12, 6), save_path=None):
    """Plot time series data"""
    plt.figure(figsize=figsize)
    if isinstance(data, dict):
        for label, series in data.items():
            plt.plot(series, label=label)
        plt.legend()
    else:
        plt.plot(data)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def rolling_split(data, window_size, step=1, return_indices=False):
    """Create rolling windows from time series data"""
    if isinstance(data, pd.DataFrame):
        data = data.values
    elif isinstance(data, list):
        data = np.array(data)
    
    windows = []
    indices = []
    
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i + window_size])
        if return_indices:
            indices.append((i, i + window_size))
    
    windows = np.array(windows)
    
    if return_indices:
        return windows, indices
    return windows

def normalize(data, method='standard', feature_range=(0, 1), return_scaler=False):
    """Normalize data using various methods"""
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    
    original_shape = data.shape
    if len(data.shape) > 1:
        data_flat = data.reshape(-1, data.shape[-1])
    else:
        data_flat = data.reshape(-1, 1)
    
    if method == 'standard':
        scaler = preprocessing.StandardScaler()
    elif method == 'minmax':
        scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
    elif method == 'robust':
        scaler = preprocessing.RobustScaler()
    elif method == 'maxabs':
        scaler = preprocessing.MaxAbsScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    normalized_data = scaler.fit_transform(data_flat)
    normalized_data = normalized_data.reshape(original_shape)
    
    if return_scaler:
        return normalized_data, scaler
    return normalized_data

def validate(model, X_val, y_val, batch_size=32, device='cpu'):
    """Validate PyTorch model and return metrics"""
    model.eval()
    val_losses = []
    predictions = []
    actuals = []
    
    # Create validation dataset
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x, training=False)
            
            # Calculate loss (MSE for validation)
            loss = torch.mean(torch.square(outputs - batch_y))
            val_losses.append(loss.item())
            
            # Store predictions and actuals
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(batch_y.cpu().numpy().flatten())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    # Avoid division by zero for MAPE
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
    
    # R-squared
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'avg_loss': np.mean(val_losses)
    }
    
    return metrics, predictions, actuals

class G:
    # preprocess
    batch_size = 64 # 128
    src_len = 20  # encoder input sequence length, the 5 is an arbitrary number
    dec_len = 1
    tgt_len = 1  # decoder input sequence length, same length as transformer output
    window_size = src_len
    mulpr_len = tgt_len
    # network
    d_model = 512
    dense_dim = 2048
    num_features = 1  # current, voltage, and soc at t minus G.window_size -> t minus 1   就输入一个差分的adjclose
    num_heads = 8
    d_k = int(d_model/num_heads)
    num_layers = 6
    dropout_rate = 0.1
    # learning_rate_scheduler
    T_i = 1
    T_mult = 2
    T_cur = 0.0
    # training
    epochs = 200 #21 should be T_i + a power of T_mult, ex) T_mult = 2 -> epochs = 2**5 + 1 = 32+1 = 33   257
    learning_rate = 0.003#0.0045
    min_learning_rate = 7e-11
    #weight_decay = 0.0 #No weight decay param in the the keras optimizers

# Train only on BTC-USD data  
filename = 'Datasets/BTC-USD.csv'
df = pd.read_csv(filename,delimiter=',',usecols=['Date','Open','High','Low','Close', 'Adj Close','Volume'])
df = df.sort_values('Date')
division_rate1 = 0.8
division_rate2 = 0.9

seq_len = G.src_len  # 20 how long of a preceeding sequence to collect for RNN
tgt = G.tgt_len
mulpre = G.mulpr_len  # how far into the future are we trying to predict?
window = G.window_size

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def get_stock_data():
    df = pd.read_csv(filename)
    df.drop(['Date', 'Close'], axis=1, inplace=True)#由于date不连续,这时候保留5维
    list = df['Adj Close']
    list1 = list.diff(1).dropna()  # list1为list的1阶差分序列,序列的序号从1开始,所以要tolist,这样序号才从0开始. 但是列表不能调用diff
    # 或者list1 = np.diff(list)[1:]
    list = list.tolist()
    list1 = list1.tolist()

    list1 = np.array(list1)#array才能reshape
    df = df.drop(0, axis=0)
    # print(df1.head())
    df['Adj Close'] = list1
    df = df.reset_index(drop=True)
    print(df.head())
    return df,list,list1

#先划分训练集测试集,再标准化归一化,避免数据泄露
def load_data(df, seq_len , mul, normalize=True):
    amount_of_features = 1  # columns是列索引,index是行索引
    data = df.values
    row1 = round(division_rate1 * data.shape[0])  #0.8  split可改动!!!!!!!#round是四舍五入,0.9可能乘出来小数  #shape[0]是result列表中子列表的个数
    row2 = round(division_rate2 * data.shape[0])  #0.9
    #训练集和测试集划分
    train = data[:int(row1), :]
    valid = data[int(row1):int(row2), :]
    test = data[int(row2): , :]

    print('train', train)
    print('valid', valid)
    print('test', test)

    # 训练集和测试集归一化
    if normalize:
        standard_scaler = preprocessing.StandardScaler()
        train = standard_scaler.fit_transform(train)
        valid = standard_scaler.transform(valid)
        test = standard_scaler.transform(test)

    print('train',train)
    print('valid', valid)
    print('test', test)
    X_train = []  # train列表中4个特征记录
    y_train = []
    X_valid = []
    y_valid = []
    X_test = []
    y_test = []
    train_samples=train.shape[0]-seq_len-mul+1
    valid_samples = valid.shape[0] - seq_len - mul + 1
    test_samples = test.shape[0] - seq_len - mul + 1
    for i in range(0,train_samples,mul):  # maximum date = lastest  date - sequence length  #index从0到极限maximum,所有天数正好被滑窗采样完
        X_train.append(train[i:i + seq_len,-2])#每个滑窗每天四个特征
        y_train.append(train[i + seq_len:i+seq_len+tgt,-2])#-1为成交量,倒数第二个才是adj close

    for i in range(0,valid_samples,mul):  # maximum date = lastest  date - sequence length  #index从0到极限maximum,所有天数正好被滑窗采样完
        X_valid.append(valid[i:i + seq_len,-2])#每个滑窗每天四个特征
        y_valid.append(valid[i+seq_len:i+seq_len+tgt,-2])#-1为成交量,倒数第二个才是adj close

    for i in range(0, test_samples,mul):  # maximum date = lastest date - sequence length  #index从0到极限maximum,所有天数正好被滑窗采样完
        X_test.append(test[i:i + seq_len, -2])  # 每个滑窗每天四个特征
        y_test.append(test[i+seq_len:i+seq_len+tgt, -2])  # -1即取最后一个特征
    # X都对应全部4特征,y都对应adj close   #train都是前百分之90,test都是后百分之10
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print('train', train.shape)
    print(train)
    print('valid', valid.shape)
    print(valid)
    print('test', test.shape)
    print(test)

    print('X_train', X_train.shape)
    print('y_train', y_train.shape)
    print('X_valid', X_valid.shape)
    print('y_valid', y_valid.shape)
    print('X_test', X_test.shape)
    print('y_test', y_test.shape)
    print('df', df)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))  # (90%maximum, seq-1 ,4) #array才能reshape
    X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  # (10%maximum, seq-1 ,4) #array才能reshape、

    print('X_train', X_train.shape)
    print('X_valid', X_valid.shape)
    print('X_test', X_test.shape)
    return X_train, y_train, X_valid, y_valid, X_test, y_test  # x是训练的数据，y是数据对应的标签,也就是说y是要预测的那一个特征!!!!!!

# Main execution
df,list,list1 = get_stock_data()
X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(df, seq_len, mulpre)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
X_valid = torch.FloatTensor(X_valid).to(device)
y_valid = torch.FloatTensor(y_valid).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.FloatTensor(y_test).to(device)


#################################

class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.dense1 = nn.Linear(G.d_model, G.dense_dim)
        self.bn1 = nn.BatchNorm1d(G.dense_dim, momentum=0.02, eps=5e-4)  # momentum in PyTorch = 1 - momentum in TF
        self.dense2 = nn.Linear(G.dense_dim, G.d_model)
        self.bn2 = nn.BatchNorm1d(G.d_model, momentum=0.05, eps=5e-4)
        
        # Initialize weights similar to TensorFlow
        nn.init.kaiming_normal_(self.dense1.weight)
        nn.init.uniform_(self.dense1.bias, 0.005, 0.08)
        nn.init.kaiming_normal_(self.dense2.weight)
        nn.init.uniform_(self.dense2.bias, 0.001, 0.01)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        x = self.dense1(x)
        x = F.relu(x, inplace=False)
        
        # Reshape for BatchNorm1d: (batch_size * seq_len, features)
        x = x.view(-1, G.dense_dim)
        x = self.bn1(x)
        x = x.view(batch_size, seq_len, G.dense_dim)
        
        x = self.dense2(x)
        
        # Reshape for BatchNorm1d: (batch_size * seq_len, features)
        x = x.view(-1, G.d_model)
        x = self.bn2(x)
        x = x.view(batch_size, seq_len, G.d_model)
        
        return x


class EncoderLayer(nn.Module):
        """
        The encoder layer is composed by a multi-head self-attention mechanism,
        followed by a simple, positionwise fully connected feed-forward network.
        This archirecture includes a residual connection around each of the two
        sub-layers, followed by batch normalization.
        """

        def __init__(self,
                     num_heads,
                     d_k,
                     dropout_rate,
                     batchnorm_eps):
            super(EncoderLayer, self).__init__()

            self.mha = nn.MultiheadAttention(
                embed_dim=G.d_model,
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=True
            )

            # feed-forward-network
            self.ffn = FullyConnected()

            self.batchnorm1 = nn.BatchNorm1d(G.d_model, momentum=0.05, eps=batchnorm_eps)
            self.batchnorm2 = nn.BatchNorm1d(G.d_model, momentum=0.05, eps=batchnorm_eps)

            self.dropout_ffn = nn.Dropout(dropout_rate)

        def forward(self, x, training=True):
            """
            Forward pass for the Encoder Layer

            Arguments:
                x -- Tensor of shape (G.batch_size, G.window_size, G.num_features)
                training -- Boolean, set to true to activate
                            the training mode for dropout layers
            Returns:
                encoder_layer_out -- Tensor of shape (G.batch_size, G.window_size, G.num_features)
            """
            batch_size, seq_len, d_model = x.shape
            
            # Self attention
            attn_output, _ = self.mha(x, x, x)

            # Add & Norm
            out1 = x + attn_output
            out1 = out1.view(-1, G.d_model)
            out1 = self.batchnorm1(out1)
            out1 = out1.view(batch_size, seq_len, G.d_model)

            ffn_output = self.ffn(out1)

            if training:
                ffn_output = self.dropout_ffn(ffn_output)

            # Add & Norm
            encoder_layer_out = out1 + ffn_output
            encoder_layer_out = encoder_layer_out.view(-1, G.d_model)
            encoder_layer_out = self.batchnorm2(encoder_layer_out)
            encoder_layer_out = encoder_layer_out.view(batch_size, seq_len, G.d_model)
            
            return encoder_layer_out


class Encoder(nn.Module):
    """
    The entire Encoder starts by passing the input to an embedding layer
    and using positional encoding to then pass the output through a stack of
    encoder Layers

    """

    def __init__(self,
                 num_layers=G.num_layers,
                 num_heads=G.num_heads,
                 num_features=G.num_features,
                 d_model=G.d_model,
                 d_k=G.d_k,
                 dense_dim=G.dense_dim,
                 maximum_position_encoding=G.src_len,
                 dropout_rate=G.dropout_rate,
                 batchnorm_eps=1e-4):
        super(Encoder, self).__init__()

        self.num_layers = num_layers

        # linear input layer
        self.lin_input = nn.Linear(num_features, d_model)

        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model).to(device)

        self.enc_layers = nn.ModuleList([EncoderLayer(num_heads=num_heads,
                                        d_k=d_k,
                                        dropout_rate=dropout_rate,
                                        batchnorm_eps=batchnorm_eps)
                           for _ in range(self.num_layers)])

    def forward(self, x, training=True):
        """
        Forward pass for the Encoder

        Arguments:
            x -- Tensor of shape (G.batch_size, G.src_len, G.num_features)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            mask -- Boolean mask to ensure that the padding is not
                    treated as part of the input
        Returns:
            Tensor of shape (G.batch_size, G.src_len, G.dense_dim)
        """
        x = F.relu(self.lin_input(x), inplace=False)
        seq_len = x.shape[1]
        x = x + self.pos_encoding[:, :seq_len, :]

        #应该concatenate！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        return x  # (G.batch_size, G.src_len, G.dense_dim)


class DecoderLayer(nn.Module):
    """
    The decoder layer is composed by two multi-head attention blocks,
    one that takes the new input and uses self-attention, and the other
    one that combines it with the output of the encoder, followed by a
    fully connected block.
    """

    def __init__(self,
                 num_heads,
                 d_k,
                 dropout_rate,
                 batchnorm_eps):
        super(DecoderLayer, self).__init__()

        self.mha1 = nn.MultiheadAttention(
            embed_dim=G.d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.mha2 = nn.MultiheadAttention(
            embed_dim=G.d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.ffn = FullyConnected()

        self.batchnorm1 = nn.BatchNorm1d(G.d_model, momentum=0.05, eps=batchnorm_eps)
        self.batchnorm2 = nn.BatchNorm1d(G.d_model, momentum=0.05, eps=batchnorm_eps)
        self.batchnorm3 = nn.BatchNorm1d(G.d_model, momentum=0.05, eps=batchnorm_eps)

        self.dropout_ffn = nn.Dropout(dropout_rate)

    def forward(self, y, enc_output, dec_ahead_mask, enc_memory_mask, training=True):
        """
        Forward pass for the Decoder Layer

        Arguments:
            y -- Tensor of shape (G.batch_size, G.tgt_len, 1) #the soc values for the batches
            enc_output --  Tensor of shape(G.batch_size, G.num_features)
            training -- Boolean, set to true to activate
                        the training mode for dropout and batchnorm layers
        Returns:
            out3 -- Tensor of shape (G.batch_size, G.tgt_len, 1)
        """
        batch_size, seq_len, d_model = y.shape

        # BLOCK 1
        # Dropout will be applied during training only
        mult_attn_out1, _ = self.mha1(y, y, y, attn_mask=dec_ahead_mask)

        Q1 = y + mult_attn_out1
        Q1 = Q1.view(-1, G.d_model)
        Q1 = self.batchnorm1(Q1)
        Q1 = Q1.view(batch_size, seq_len, G.d_model)

        # BLOCK 2
        # calculate self-attention using the Q from the first block and K and V from the encoder output.
        # Dropout will be applied during training
        mult_attn_out2, _ = self.mha2(Q1, enc_output, enc_output, attn_mask=enc_memory_mask)

        mult_attn_out2 = mult_attn_out1 + mult_attn_out2
        mult_attn_out2 = mult_attn_out2.view(-1, G.d_model)
        mult_attn_out2 = self.batchnorm2(mult_attn_out2)
        mult_attn_out2 = mult_attn_out2.view(batch_size, seq_len, G.d_model)

        # BLOCK 3
        # pass the output of the second block through a ffn
        ffn_output = self.ffn(mult_attn_out2)

        # apply a dropout layer to the ffn output
        if training:
            ffn_output = self.dropout_ffn(ffn_output)

        out3 = ffn_output + mult_attn_out2
        out3 = out3.view(-1, G.d_model)
        out3 = self.batchnorm3(out3)
        out3 = out3.view(batch_size, seq_len, G.d_model)
        
        return out3


class Decoder(nn.Module):
    """

    """

    def __init__(self,
                 num_layers=G.num_layers,
                 num_heads=G.num_heads,
                 num_features=G.num_features,
                 d_model=G.d_model,
                 d_k=G.d_k,
                 dense_dim=G.dense_dim,
                 target_size=G.num_features,
                 maximum_position_encoding=G.dec_len,
                 dropout_rate=G.dropout_rate,
                 batchnorm_eps=1e-5):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model).to(device)

        # linear input layer
        self.lin_input = nn.Linear(num_features, d_model)

        self.dec_layers = nn.ModuleList([DecoderLayer(num_heads=num_heads,
                                        d_k=d_k,
                                        dropout_rate=dropout_rate,
                                        batchnorm_eps=batchnorm_eps
                                        )
                           for _ in range(self.num_layers)])
        # look_ahead_masks for decoder:
        self.dec_ahead_mask = create_look_ahead_mask(G.dec_len, G.dec_len).to(device)
        self.enc_memory_mask = create_look_ahead_mask(G.dec_len, G.src_len).to(device)

    def forward(self, y, enc_output, training=True):
        """
        Forward  pass for the Decoder

        Arguments:
            y -- Tensor of shape (G.batch_size, G.tgt_len, G.dense_dim) #the final SOC values in the batches
            enc_output --  Tensor of shape(G.batch_size, G.src_len, G.dense_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
        Returns:
            y -- Tensor of shape (G.batch_size, G.tgt_len, 1)
        """
        y = F.relu(self.lin_input(y), inplace=False)  # maps to dense_dim, the dimension of all the sublayer outputs.

        dec_len = y.shape[1]
        # print('dec_len',dec_len)
        y = y + self.pos_encoding[:, :dec_len, :]

        # use a for loop to pass y through a stack of decoder layers and update attention_weights
        for i in range(self.num_layers):
            # pass y and the encoder output through a stack of decoder layers and save attention weights
            y = self.dec_layers[i](y,
                                   enc_output,
                                   self.dec_ahead_mask,
                                   self.enc_memory_mask,
                                   training)

        # print('y.shape', y.shape)
        return y


class Transformer(nn.Module):
    """
    Complete transformer with an Encoder and a Decoder
    """

    def __init__(self,
                 num_layers=G.num_layers,
                 num_heads=G.num_heads,
                 dense_dim=G.dense_dim,
                 src_len=G.src_len,
                 dec_len = G.dec_len,
                 tgt_len=G.tgt_len,
                 max_positional_encoding_input=G.src_len,
                 max_positional_encoding_target=G.tgt_len):
        super(Transformer, self).__init__()

        self.tgt_len = tgt_len
        self.dec_len = dec_len
        self.src_len = src_len

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.linear_map = nn.Sequential(
            nn.Linear(G.d_model, dense_dim),
            nn.ReLU(),
            nn.BatchNorm1d(dense_dim, momentum=0.03, eps=5e-4),
            nn.Linear(dense_dim, 1)
        )
        
        # Initialize weights
        for layer in self.linear_map:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.out_features == dense_dim:
                    nn.init.uniform_(layer.bias, 0.001, 0.02)

    def forward(self, x, training=True):
        """
        Forward pass for the entire Transformer
        Arguments:
            x -- Tensor of shape (G.batch_size, G.window_size, G.num_features)
                 An array of the windowed voltage, current and soc data
            training -- Boolean, set to true to activate
                        the training mode for dropout and batchnorm layers
        Returns:
            final_output -- SOC prediction at time t

        """
        enc_input = x[:, :self.src_len, :]   # (G.batch_size, G.src_len, G.num_features)
        dec_input = x[:, -self.dec_len:, ]  # only want the SOC thats why -1 is there!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # print(type(dec_input))
        # print('dec_input.shape',dec_input.shape)

        enc_output = self.encoder(enc_input, training)  # (G.batch_size, G.src_len, G.num_features)
        # print('enc_output.shape', enc_output.shape)

        dec_output = self.decoder(dec_input, enc_output, training)
        # print('dec_output.shape', dec_output.shape)
        # (G.batch_size, G.tgt_len, 32)

        batch_size, seq_len, d_model = dec_output.shape
        
        # Apply linear mapping
        dec_output_flat = dec_output.view(-1, G.d_model)
        final_output = self.linear_map[0](dec_output_flat)  # Linear layer
        final_output = self.linear_map[1](final_output)     # ReLU
        final_output = self.linear_map[2](final_output)     # BatchNorm
        final_output = self.linear_map[3](final_output)     # Final Linear
        final_output = final_output.view(batch_size, seq_len, 1)

        # print('final_output.shape', final_output.shape)
        
        # Transpose and apply final dense layer (equivalent to TF operations)
        final_output = final_output.transpose(1, 2)  # (batch_size, 1, seq_len)
        final_dense = nn.Linear(seq_len, G.tgt_len).to(device)
        final_output = final_dense(final_output)
        final_output = final_output.transpose(1, 2)  # (batch_size, tgt_len, 1)
        
        # print('final_output.shape', final_output.shape)
        return final_output

def calculate_accuracy(pre, real):
    print('pre.shape', pre.shape)
    print(pre)

    print('real.shape', real.shape)
    print(real)
    # MAPE = np.mean(np.abs((pre - real) / real))
    MAPE = sklearn.metrics.mean_absolute_percentage_error(real,pre)
    #MAPE = calculate_MAPE(pre,real)
    RMSE = np.sqrt(np.mean(np.square(pre - real)))
    MAE = np.mean(np.abs(pre - real))
    R2 = r2_score(pre, real)
    dict = {'MAPE': [MAPE], 'RMSE': [RMSE], 'MAE': [MAE], 'R2': [R2]}
    df = pd.DataFrame(dict)
    print('最终的准确率和指标如下\n',df)
    return df

def up_down_accuracy_loss(real, pre):
    '''products = []

    print('tf.shape',tf.shape(real))
    print('pre.shape',pre.shape)
    for i in tf.range(tf.shape(real)[0]):
        products.append(real[i] *  pre[i])
    accuracy = (sum([int(x > 0) for x in products])) / len(products)
    return accuracy'''
    # print('real.shape', real.shape)
    # print('pre.shape', pre.shape)
    mse = torch.mean(torch.square(pre - real))
    # print('mse！！！！！', mse.item())
    # print('real666.shape', real.shape)
    # print('pre666.shape', pre.shape)#real666.shape (None, 3)  pre666.shape (None, 3)
    # print('real666', real)
    # print('pre666', pre)
    accu = torch.multiply(real, pre)#矩阵点积，不是乘法，得出正负，正的就是趋势预测正确
    accu = F.relu(accu, inplace=False)#relu(x) = max(0,x)
    accu = torch.sign(accu)#正数变1，0不变
    accu = torch.mean(accu)#取平均
    # print('accu！！！！！', accu.item())#准确率，0.x
    '''result = tf.compat.v1.Session().run(result)

    print('resultnumpy', result)
    accuracy = (sum([int(x > 0) for x in result]))
    print('loss666', tf.math.reduce_mean(tf.square(real - pre)))'''
    accu = 1 - accu#loss越小越好，所以1-准确率S
    #loss = mse + accu * 10 #mse个位数，accu 0.x
    loss = accu * pow(10, floor(math.log(abs(mse.item()), 10))) + mse
    return loss#个位数


def denormalize(normalized_value):
    df = pd.read_csv(filename, usecols=['Adj Close'])
    list = df['Adj Close']
    list1 = list.diff(1).dropna()  # list1为list的1阶差分序列,序列的序号从1开始,所以要tolist,这样序号才从0开始. 但是列表不能调用diff
    # 或者list1 = np.diff(list)[1:]
    list1 = list1.tolist()
    list1 = np.array(list1)  # array才能reshape
    df1 = df.drop(0, axis=0)
    df1['Adj Close'] = list1
    df1 = df1.reset_index(drop=True)#index从0开始
    print(df.head())
    print(df1.head())
    data = df.values
    data1 = df1.values
    row1 = round(division_rate1 * list1.shape[0])
    row2 = round(division_rate2 * list1.shape[0])
    
    # 训练集和测试集划分
    train = data1[ :int(row1), :]
    test = data1[int(row2): , :]
    test = test.reshape(-1, 1)#取原来没有归一化的adj数据作为样本
    standard_scaler = preprocessing.StandardScaler()
    m = standard_scaler.fit_transform(train)  # 利用m对data进行归一化，并储存df的归一化参数. 用训练集的归一化参数来反归一化y_test和预测值


    '反归一化'
    normalized_value = normalized_value.reshape(-1, 1)
    new = standard_scaler.inverse_transform(normalized_value)#利用m对normalized_value进行反归一化
    print('new',new.shape)

    length = y_test.shape[0]
    residual = data[int(row2) + seq_len : int(row2) + seq_len +  mulpre * length, : ]#差分残差从test的seq-1序号天开始到test的倒数第二天,预测加上前一天的残差对应test[seq:]反归一的真实值,注意y_test和预测值是一致的
    print('residual', residual.shape)

    sum = new + residual
    '归一化'
    '''m = min_max_scaler.fit_transform(train)  # 利用m对train进行归一化，并储存df的归一化参数!!
    new = min_max_scaler.transform(test)  # 利用m对test也进行归一化,注意这里是transform不能是fit_transform!!!1'''
    return new,sum#new是差分预测值，sum是没差分的预测值

# Create model and training setup
model = Transformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=G.learning_rate)

# Create DataLoaders
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=G.batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=G.batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=G.batch_size, shuffle=False)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Enable anomaly detection to find the in-place operation
torch.autograd.set_detect_anomaly(True)

# Training loop
model.train()
for epoch in range(G.epochs):
    total_loss = 0
    num_batches = 0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch_x, training=True)
        
        # Calculate loss
        loss = up_down_accuracy_loss(batch_y, predictions)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    # Validation
    if epoch % 10 == 0:
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                val_predictions = model(batch_x, training=False)
                val_loss += up_down_accuracy_loss(batch_y, val_predictions).item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        print(f'Epoch {epoch}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        model.train()

# Testing
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        predictions = model(batch_x, training=False)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

# Concatenate all predictions and targets
all_predictions = np.concatenate(all_predictions, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# Denormalize predictions and calculate accuracy
pred_denorm, pred_sum = denormalize(all_predictions)
target_denorm, target_sum = denormalize(all_targets)

# Calculate and display accuracy metrics
accuracy_df = calculate_accuracy(pred_sum.flatten(), target_sum.flatten())

# Save results
results_df = pd.DataFrame({
    'Predictions': pred_sum.flatten(),
    'Actual': target_sum.flatten()
})
results_df.to_csv('BTC-USD_Galformer_predictions.csv', index=False)
print("Results saved to BTC-USD_Galformer_predictions.csv")

# Plot results if data_plot function is available
try:
    data_plot(pred_sum.flatten(), target_sum.flatten(), 'BTC-USD Galformer Predictions')
except:
    print("Plotting function not available, skipping visualization")

print("Training and evaluation completed!")
