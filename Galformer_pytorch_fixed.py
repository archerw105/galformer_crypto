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
import yaml
import sys
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Load configuration from YAML file
print("Loading config.yaml...", flush=True)
try:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print("Config loaded successfully!", flush=True)
except Exception as e:
    print(f"Error loading config: {e}", flush=True)
    raise

# Create experiment directory structure
experiment_name = config['experiment']['name']
experiment_dir = f"runs/{experiment_name}"
os.makedirs(experiment_dir, exist_ok=True)
print(f"Created experiment directory: {experiment_dir}", flush=True)

# Update output paths to use experiment directory
config['output']['model_path'] = os.path.join(experiment_dir, config['output']['model_path'])
config['output']['predictions_path'] = os.path.join(experiment_dir, config['output']['predictions_path'])
config['output']['plot_path'] = os.path.join(experiment_dir, config['output']['plot_path'])

print(f"Outputs will be saved to: {experiment_dir}", flush=True)

def load_model(model_path, device):
    """
    Load a pre-trained Galformer model from checkpoint
    
    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load the model on
        
    Returns:
        tuple: (model, optimizer, scaler, saved_config)
    """
    print(f"Loading model from {model_path}...", flush=True)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract saved configuration
    saved_config = checkpoint.get('config', None)
    if saved_config is None:
        print("Warning: No config found in checkpoint, using current config", flush=True)
        saved_config = config
    
    # Create model with saved configuration
    model = Transformer().to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create optimizer and load its state if available
    optimizer = torch.optim.Adam(model.parameters(), lr=saved_config.get('training', {}).get('learning_rate', 0.001))
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scaler if available
    scaler = checkpoint.get('scaler', None)
    
    print("Model loaded successfully!", flush=True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}", flush=True)
    
    return model, optimizer, scaler, saved_config

class G:
    # Data preprocessing
    batch_size = config['training']['batch_size']
    src_len = config['sequence']['src_len']
    dec_len = config['sequence']['dec_len']
    tgt_len = config['sequence']['tgt_len']
    window_size = config['sequence']['window_size']
    mulpr_len = config['sequence']['mulpr_len']
    
    # Network architecture
    d_model = config['model']['d_model']
    dense_dim = config['model']['dense_dim']
    num_features = config['model']['num_features']
    num_heads = config['model']['num_heads']
    d_k = int(d_model/num_heads)
    num_layers = config['model']['num_layers']
    dropout_rate = config['model']['dropout_rate']
    
    # Learning rate scheduler
    T_i = config['scheduler']['T_i']
    T_mult = config['scheduler']['T_mult']
    T_cur = config['scheduler']['T_cur']
    
    # Training parameters
    epochs = config['training']['epochs']
    learning_rate = config['training']['learning_rate']
    min_learning_rate = config['training']['min_learning_rate']

# Train only on BTC-USD data  
filename = config['data']['filename']
df = pd.read_csv(filename,delimiter=',',usecols=['Date','Open','High','Low','Close', 'Adj Close','Volume'])
df = df.sort_values('Date')
division_rate1 = config['data']['division_rate1']
division_rate2 = config['data']['division_rate2']

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
    print(df.head(), flush=True)
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

    print('train', train, flush=True)
    print('valid', valid, flush=True)
    print('test', test, flush=True)

    # 训练集和测试集归一化
    if normalize:
        standard_scaler = preprocessing.StandardScaler()
        train = standard_scaler.fit_transform(train)
        valid = standard_scaler.transform(valid)
        test = standard_scaler.transform(test)
    else:
        standard_scaler = None

    print('train',train, flush=True)
    print('valid', valid, flush=True)
    print('test', test, flush=True)
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
    print('train', train.shape, flush=True)
    print(train, flush=True)
    print('valid', valid.shape, flush=True)
    print(valid, flush=True)
    print('test', test.shape, flush=True)
    print(test, flush=True)

    print('X_train', X_train.shape, flush=True)
    print('y_train', y_train.shape, flush=True)
    print('X_valid', X_valid.shape, flush=True)
    print('y_valid', y_valid.shape, flush=True)
    print('X_test', X_test.shape, flush=True)
    print('y_test', y_test.shape, flush=True)
    print('df', df, flush=True)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))  # (90%maximum, seq-1 ,4) #array才能reshape
    X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  # (10%maximum, seq-1 ,4) #array才能reshape、

    print('X_train', X_train.shape, flush=True)
    print('X_valid', X_valid.shape, flush=True)
    print('X_test', X_test.shape, flush=True)
    return X_train, y_train, X_valid, y_valid, X_test, y_test, standard_scaler  # x是训练的数据，y是数据对应的标签,也就是说y是要预测的那一个特征!!!!!!



#################################

class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.dense1 = nn.Linear(G.d_model, G.dense_dim)
        self.bn1 = nn.BatchNorm1d(G.dense_dim, momentum=0.98, eps=5e-4)  # Match TF momentum exactly
        self.dense2 = nn.Linear(G.dense_dim, G.d_model)
        self.bn2 = nn.BatchNorm1d(G.d_model, momentum=0.95, eps=5e-4)
        
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
        x = x.reshape(-1, G.dense_dim)
        x = self.bn1(x)
        x = x.reshape(batch_size, seq_len, G.dense_dim)
        
        x = self.dense2(x)
        
        # Reshape for BatchNorm1d: (batch_size * seq_len, features)
        x = x.reshape(-1, G.d_model)
        x = self.bn2(x)
        x = x.reshape(batch_size, seq_len, G.d_model)
        
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

            self.batchnorm1 = nn.BatchNorm1d(G.d_model, momentum=0.95, eps=batchnorm_eps)
            self.batchnorm2 = nn.BatchNorm1d(G.d_model, momentum=0.95, eps=batchnorm_eps)

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
            out1 = out1.reshape(-1, G.d_model)
            out1 = self.batchnorm1(out1)
            out1 = out1.reshape(batch_size, seq_len, G.d_model)

            ffn_output = self.ffn(out1)

            if training:
                ffn_output = self.dropout_ffn(ffn_output)

            # Add & Norm
            encoder_layer_out = out1 + ffn_output
            encoder_layer_out = encoder_layer_out.reshape(-1, G.d_model)
            encoder_layer_out = self.batchnorm2(encoder_layer_out)
            encoder_layer_out = encoder_layer_out.reshape(batch_size, seq_len, G.d_model)
            
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

        self.batchnorm1 = nn.BatchNorm1d(G.d_model, momentum=0.95, eps=batchnorm_eps)
        self.batchnorm2 = nn.BatchNorm1d(G.d_model, momentum=0.95, eps=batchnorm_eps)
        self.batchnorm3 = nn.BatchNorm1d(G.d_model, momentum=0.95, eps=batchnorm_eps)

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
        Q1 = Q1.reshape(-1, G.d_model)
        Q1 = self.batchnorm1(Q1)
        Q1 = Q1.reshape(batch_size, seq_len, G.d_model)

        # BLOCK 2
        # calculate self-attention using the Q from the first block and K and V from the encoder output.
        # Dropout will be applied during training
        mult_attn_out2, _ = self.mha2(Q1, enc_output, enc_output, attn_mask=enc_memory_mask)

        mult_attn_out2 = mult_attn_out1 + mult_attn_out2
        mult_attn_out2 = mult_attn_out2.reshape(-1, G.d_model)
        mult_attn_out2 = self.batchnorm2(mult_attn_out2)
        mult_attn_out2 = mult_attn_out2.reshape(batch_size, seq_len, G.d_model)

        # BLOCK 3
        # pass the output of the second block through a ffn
        ffn_output = self.ffn(mult_attn_out2)

        # apply a dropout layer to the ffn output
        if training:
            ffn_output = self.dropout_ffn(ffn_output)

        out3 = ffn_output + mult_attn_out2
        out3 = out3.reshape(-1, G.d_model)
        out3 = self.batchnorm3(out3)
        out3 = out3.reshape(batch_size, seq_len, G.d_model)
        
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
            nn.BatchNorm1d(dense_dim, momentum=0.97, eps=5e-4),
            nn.Linear(dense_dim, 1)
        )
        
        # Final dense layer for sequence length transformation
        # Note: This will be dynamically sized based on actual sequence length in forward pass
        # We'll initialize it with a placeholder and recreate if needed
        self.final_dense = None
        
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
        dec_output_flat = dec_output.reshape(-1, G.d_model)
        final_output = self.linear_map[0](dec_output_flat)  # Linear layer
        final_output = self.linear_map[1](final_output)     # ReLU
        final_output = self.linear_map[2](final_output)     # BatchNorm
        final_output = self.linear_map[3](final_output)     # Final Linear
        final_output = final_output.reshape(batch_size, seq_len, 1)

        # print('final_output.shape', final_output.shape)
        
        # Transpose and apply final dense layer (equivalent to TF operations)
        final_output = final_output.transpose(1, 2)  # (batch_size, 1, seq_len)
        
        # Create or reuse final dense layer with proper input size
        if self.final_dense is None or self.final_dense.in_features != seq_len:
            self.final_dense = nn.Linear(seq_len, G.tgt_len).to(device)
            # Initialize the weights properly
            nn.init.kaiming_normal_(self.final_dense.weight)
            nn.init.uniform_(self.final_dense.bias, 0.001, 0.02)
        
        final_output = self.final_dense(final_output)
        final_output = final_output.transpose(1, 2)  # (batch_size, tgt_len, 1)
        
        # print('final_output.shape', final_output.shape)
        return final_output

def calculate_accuracy(pre, real):
    # print('pre.shape', pre.shape, flush=True)
    # print('real.shape', real.shape, flush=True)
    
    # Overall metrics
    MAPE = sklearn.metrics.mean_absolute_percentage_error(real, pre)
    RMSE = np.sqrt(np.mean(np.square(pre - real)))
    MAE = np.mean(np.abs(pre - real))
    R2 = r2_score(real, pre)
    
    dict = {'MAPE': [MAPE], 'RMSE': [RMSE], 'MAE': [MAE], 'R2': [R2]}
    df = pd.DataFrame(dict)
    print('Overall accuracy metrics:\n', df, flush=True)
    return df

def calculate_multi_day_accuracy(predictions, targets):
    """
    Calculate accuracy metrics for multi-day predictions
    predictions: shape (n_samples, n_days)
    targets: shape (n_samples, n_days)
    """
    print(f'Multi-day predictions shape: {predictions.shape}', flush=True)
    print(f'Multi-day targets shape: {targets.shape}', flush=True)
    
    n_days = predictions.shape[1] if len(predictions.shape) > 1 else 1
    
    if n_days == 1:
        # Single day prediction - use existing function
        return calculate_accuracy(predictions.flatten(), targets.flatten())
    
    # Multi-day predictions
    results = {}
    
    for day in range(n_days):
        day_pred = predictions[:, day] if len(predictions.shape) > 1 else predictions
        day_target = targets[:, day] if len(targets.shape) > 1 else targets
        
        mape = sklearn.metrics.mean_absolute_percentage_error(day_target, day_pred)
        rmse = np.sqrt(np.mean(np.square(day_pred - day_target)))
        mae = np.mean(np.abs(day_pred - day_target))
        r2 = r2_score(day_target, day_pred)
        
        results[f'Day_{day+1}'] = {
            'MAPE': mape,
            'RMSE': rmse, 
            'MAE': mae,
            'R2': r2
        }
        
        print(f'Day {day+1} metrics - MAPE: {mape:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}', flush=True)
    
    # Overall metrics across all days
    overall_mape = sklearn.metrics.mean_absolute_percentage_error(targets.flatten(), predictions.flatten())
    overall_rmse = np.sqrt(np.mean(np.square(predictions.flatten() - targets.flatten())))
    overall_mae = np.mean(np.abs(predictions.flatten() - targets.flatten()))
    overall_r2 = r2_score(targets.flatten(), predictions.flatten())
    
    results['Overall'] = {
        'MAPE': overall_mape,
        'RMSE': overall_rmse,
        'MAE': overall_mae, 
        'R2': overall_r2
    }
    
    print(f'Overall metrics - MAPE: {overall_mape:.4f}, RMSE: {overall_rmse:.4f}, MAE: {overall_mae:.4f}, R2: {overall_r2:.4f}', flush=True)
    
    # Convert to DataFrame for easy viewing
    results_df = pd.DataFrame(results).T
    print('Multi-day prediction accuracy metrics:\n', results_df, flush=True)
    
    return results_df

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
    
    # Flatten tensors to match original TensorFlow behavior
    real_flat = real.view(-1)
    pre_flat = pre.view(-1)
    
    # print('real_flat.shape', real_flat.shape)
    # print('pre_flat.shape', pre_flat.shape)
    
    mse = torch.mean(torch.square(pre_flat - real_flat))
    # print('mse！！！！！', mse.item())
    # print('real666.shape', real_flat.shape)
    # print('pre666.shape', pre_flat.shape)#real666.shape (None, 3)  pre666.shape (None, 3)
    # print('real666', real_flat)
    # print('pre666', pre_flat)
    accu = torch.multiply(real_flat, pre_flat)#矩阵点积，不是乘法，得出正负，正的就是趋势预测正确
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
    print(df.head(), flush=True)
    print(df1.head(), flush=True)
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
    print('new',new.shape, flush=True)

    length = y_test.shape[0]
    residual = data[int(row2) + seq_len : int(row2) + seq_len +  mulpre * length, : ]#差分残差从test的seq-1序号天开始到test的倒数第二天,预测加上前一天的残差对应test[seq:]反归一的真实值,注意y_test和预测值是一致的
    print('residual', residual.shape, flush=True)

    sum = new + residual
    '归一化'
    '''m = min_max_scaler.fit_transform(train)  # 利用m对train进行归一化，并储存df的归一化参数!!
    new = min_max_scaler.transform(test)  # 利用m对test也进行归一化,注意这里是transform不能是fit_transform!!!1'''
    return new,sum#new是差分预测值，sum是没差分的预测值

def calculate_roi(predictions, actual_prices, initial_capital=10000, allow_shorting=True):
    """
    Calculate ROI based on trading strategy using predictions
    
    Strategy: 
    - Buy (long) when prediction > current price
    - Sell (short) when prediction < current price (if shorting allowed)
    - Close positions when prediction reverses
    
    Args:
        predictions: Array of predicted prices
        actual_prices: Array of actual prices
        initial_capital: Starting capital amount
        allow_shorting: If True, allows short positions; if False, only long positions
    """
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long position, -1 = short position
    shares = 0
    trades = []
    portfolio_values = [initial_capital]
    
    for i in range(1, len(predictions)):
        current_price = actual_prices[i-1]
        predicted_price = predictions[i]
        next_actual_price = actual_prices[i]
        
        # Trading decision based on prediction
        if predicted_price > current_price:
            if position == 0:
                # Buy signal - go long
                shares = capital / current_price
                capital = 0
                position = 1
                trades.append(('BUY', current_price, shares))
            elif position == -1 and allow_shorting:
                # Close short position and go long
                # First close short: pay back borrowed shares
                capital = capital - (shares * current_price)  # Cost to buy back shares
                trades.append(('COVER', current_price, shares))
                # Then go long with remaining capital
                if capital > 0:
                    shares = capital / current_price
                    capital = 0
                    position = 1
                    trades.append(('BUY', current_price, shares))
                else:
                    # Not enough capital to go long after covering short
                    shares = 0
                    position = 0
                    
        elif predicted_price < current_price:
            if position == 0 and allow_shorting:
                # Short signal - go short
                shares = initial_capital / current_price  # Borrow shares worth initial capital
                capital = initial_capital + (shares * current_price)  # Capital + proceeds from short sale
                position = -1
                trades.append(('SHORT', current_price, shares))
            elif position == 1:
                # Close long position
                capital = shares * current_price
                position = 0
                trades.append(('SELL', current_price, shares))
                shares = 0
                # If shorting allowed, go short after closing long
                if allow_shorting:
                    shares = capital / current_price
                    capital = capital + (shares * current_price)
                    position = -1
                    trades.append(('SHORT', current_price, shares))
        
        # Calculate portfolio value
        if position == 1:  # Long position
            portfolio_value = shares * next_actual_price
        elif position == -1:  # Short position
            # Portfolio value = capital - (current value of borrowed shares)
            portfolio_value = capital - (shares * next_actual_price)
        else:  # No position
            portfolio_value = capital
            
        portfolio_values.append(portfolio_value)
    
    # Close any remaining position
    if position == 1:
        capital = shares * actual_prices[-1]
        trades.append(('SELL', actual_prices[-1], shares))
    elif position == -1:
        capital = capital - (shares * actual_prices[-1])
        trades.append(('COVER', actual_prices[-1], shares))
    
    final_value = capital
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    # Calculate buy and hold return for comparison
    buy_hold_shares = initial_capital / actual_prices[0]
    buy_hold_final = buy_hold_shares * actual_prices[-1]
    buy_hold_return = (buy_hold_final - initial_capital) / initial_capital * 100
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return_pct': total_return,
        'buy_hold_return_pct': buy_hold_return,
        'num_trades': len(trades),
        'trades': trades,
        'portfolio_values': portfolio_values
    }


def plot_roi_analysis(roi_results, predictions, actual_prices, save_path=None):
    """Plot ROI analysis results"""
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Portfolio Value Over Time
    initial_capital = roi_results['initial_capital']
    buy_hold_shares = initial_capital / actual_prices[0]
    buy_hold_portfolio = [buy_hold_shares * price for price in actual_prices]
    
    ax1.plot(roi_results['portfolio_values'], label='Strategy Portfolio', linewidth=2, color='blue')
    ax1.plot(buy_hold_portfolio, label='Buy & Hold', linewidth=2, color='orange', alpha=0.8)
    ax1.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Capital', alpha=0.7)
    ax1.set_title('Portfolio Value: Strategy vs Buy & Hold')
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predictions vs Actual Prices
    time_idx = range(len(predictions))
    ax2.plot(time_idx, predictions, label='Predictions', alpha=0.7)
    ax2.plot(time_idx, actual_prices, label='Actual Prices', alpha=0.7)
    ax2.set_title('Price Predictions vs Actual')
    ax2.set_xlabel('Time Period')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Returns Comparison
    strategy_return = roi_results['total_return_pct']
    buy_hold_return = roi_results['buy_hold_return_pct']
    
    returns = [strategy_return, buy_hold_return]
    labels = ['Strategy', 'Buy & Hold']
    colors = ['blue', 'orange']
    
    bars = ax3.bar(labels, returns, color=colors, alpha=0.7)
    ax3.set_title('Total Returns Comparison')
    ax3.set_ylabel('Return (%)')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, return_val in zip(bars, returns):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{return_val:.1f}%', ha='center', va='bottom')
    
    # Plot 4: Trading Activity
    if len(roi_results['portfolio_values']) > 1:
        daily_returns = np.diff(roi_results['portfolio_values']) / roi_results['portfolio_values'][:-1] * 100
        ax4.hist(daily_returns, bins=20, alpha=0.7, edgecolor='black')
        ax4.axvline(x=np.mean(daily_returns), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(daily_returns):.3f}%')
        ax4.set_title('Daily Returns Distribution')
        ax4.set_xlabel('Daily Return (%)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROI analysis plot saved to {save_path}")
    
    plt.show()


def analyze_trading_performance(predictions, actual_prices, initial_capital=10000, save_path=None):
    """
    Complete trading performance analysis
    """
    print("\n" + "="*60)
    print("📊 TRADING PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Calculate ROI
    roi_results = calculate_roi(predictions, actual_prices, initial_capital)
    
    # Print results
    print(f"\n💰 FINANCIAL RESULTS")
    print(f"Initial Capital: ${roi_results['initial_capital']:,.2f}")
    print(f"Final Portfolio Value: ${roi_results['final_value']:,.2f}")
    print(f"Strategy Return: {roi_results['total_return_pct']:.2f}%")
    print(f"Buy & Hold Return: {roi_results['buy_hold_return_pct']:.2f}%")
    
    excess_return = roi_results['total_return_pct'] - roi_results['buy_hold_return_pct']
    print(f"Excess Return: {excess_return:.2f}% {'✅' if excess_return > 0 else '❌'}")
    
    print(f"\n📈 TRADING STATISTICS")
    print(f"Number of Trades: {roi_results['num_trades']}")
    
    if len(roi_results['portfolio_values']) > 1:
        daily_returns = np.diff(roi_results['portfolio_values']) / roi_results['portfolio_values'][:-1]
        win_rate = np.sum(daily_returns > 0) / len(daily_returns) * 100
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average Daily Return: {np.mean(daily_returns) * 100:.3f}%")
    
    print(f"\n🔄 RECENT TRADES (Last 5)")
    recent_trades = roi_results['trades'][-5:] if len(roi_results['trades']) >= 5 else roi_results['trades']
    for trade in recent_trades:
        action, price, shares = trade
        print(f"  {action}: {shares:.2f} shares @ ${price:.2f}")
    
    # Create visualization
    plot_roi_analysis(roi_results, predictions, actual_prices, save_path)
    
    return roi_results


if __name__ == "__main__":
    print("Starting Galformer script...", flush=True)

    
    if torch.cuda.is_available():
        print(f'Using GPU: {torch.cuda.get_device_name(0)}', flush=True)
        print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB', flush=True)
    else:
        print("Using CPU", flush=True)

    # Main execution
    df,list,list1 = get_stock_data()
    X_train, y_train, X_valid, y_valid, X_test, y_test, standard_scaler = load_data(df, seq_len, mulpre)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_valid = torch.FloatTensor(X_valid).to(device)
    y_valid = torch.FloatTensor(y_valid).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    # Create model and training setup
    load_model_path = config['training'].get('load_model_path', None)
    skip_training = config['training'].get('skip_training', False)
    
    if load_model_path and os.path.exists(load_model_path):
        # Load pre-trained model
        model, optimizer, loaded_scaler, saved_config = load_model(load_model_path, device)
        
        # Use loaded scaler if available, otherwise use the current one
        if loaded_scaler is not None:
            print("Using scaler from loaded model", flush=True)
            standard_scaler = loaded_scaler
        else:
            print("No scaler found in loaded model, using current scaler", flush=True)
            
        # Update G class with loaded config if needed
        if saved_config != config:
            print("Note: Loaded model was trained with different configuration", flush=True)
            
    else:
        # Create new model
        if load_model_path:
            print(f"Warning: Model path '{load_model_path}' not found, creating new model", flush=True)
        model = Transformer().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=G.learning_rate)

    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=G.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=G.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=G.batch_size, shuffle=False)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}", flush=True)

    # Training loop (skip if requested)
    if not skip_training:
        print("Starting training...", flush=True)
        
        # Enable anomaly detection to find the in-place operation
        torch.autograd.set_detect_anomaly(True)

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
                print(f'Epoch {epoch}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}', flush=True)
                model.train()
        
        print("Training completed!", flush=True)
    else:
        print("Skipping training (using loaded model)", flush=True)

    # Testing - Ensure model is in evaluation mode for deterministic inference
    model.eval()  # Disables dropout and uses fixed BatchNorm statistics
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

    # Calculate and display accuracy metrics for multi-day predictions
    accuracy_df = calculate_multi_day_accuracy(pred_sum, target_sum)

    # Save results with individual day predictions
    if pred_sum.shape[1] > 1:  # Multi-day predictions
        results_data = {}
        for day in range(pred_sum.shape[1]):
            results_data[f'Prediction_Day_{day+1}'] = pred_sum[:, day]
            results_data[f'Actual_Day_{day+1}'] = target_sum[:, day]
        results_df = pd.DataFrame(results_data)
    else:  # Single day predictions
        results_df = pd.DataFrame({
            'Predictions': pred_sum.flatten(),
            'Actual': target_sum.flatten()
        })

    

    results_df.to_csv(config['output']['predictions_path'], index=False)
    print(f"Results saved to {config['output']['predictions_path']}", flush=True)
    print(f"Saved {pred_sum.shape[0]} samples with {pred_sum.shape[1]}-day predictions", flush=True)

    # Plot results if data_plot function is available
    try:
        data_plot({'Predictions': pred_sum.flatten(), 'Actual': target_sum.flatten()}, 
                f"{config['experiment']['name']} {pred_sum.shape[1]}-Day Predictions", save_path=config['output']['plot_path'])
    except:
        print("Plotting function not available, skipping visualization", flush=True)

    # Save the trained model (only if we trained)
    if not skip_training:
        model_save_path = config['output']['model_path']
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler': standard_scaler,  # Save the fitted scaler
            'config': config  # Save the entire config instead of manual specification
        }, model_save_path)

        print(f"Model saved to {model_save_path}", flush=True)
        print("Training and evaluation completed!", flush=True)
    else:
        print("Evaluation completed (model was loaded)!", flush=True)
    
    # ROI Analysis - Compare Long-Only vs Long-Short strategies
    print("\n🚀 Starting ROI Analysis...", flush=True)
    try:
        # Use the denormalized predictions and targets for ROI analysis
        if pred_sum.shape[1] == 1:  # Single day predictions
            predictions_for_roi = pred_sum.flatten()
            actuals_for_roi = target_sum.flatten()
        else:  # Multi-day predictions - use first day
            predictions_for_roi = pred_sum[:, 0]
            actuals_for_roi = target_sum[:, 0]
        
        # Run ROI analysis WITHOUT shorting (Long-Only)
        print("\n📈 Long-Only Strategy:", flush=True)
        roi_long_only = calculate_roi(predictions_for_roi, actuals_for_roi, 
                                     initial_capital=10000, allow_shorting=False)
        print(f"Long-Only Return: {roi_long_only['total_return_pct']:.2f}%", flush=True)
        print(f"Buy & Hold Return: {roi_long_only['buy_hold_return_pct']:.2f}%", flush=True)
        print(f"Number of Trades: {roi_long_only['num_trades']}", flush=True)
        
        # Run ROI analysis WITH shorting (Long-Short)
        print("\n📊 Long-Short Strategy:", flush=True)
        roi_long_short = calculate_roi(predictions_for_roi, actuals_for_roi, 
                                      initial_capital=10000, allow_shorting=True)
        print(f"Long-Short Return: {roi_long_short['total_return_pct']:.2f}%", flush=True)
        print(f"Buy & Hold Return: {roi_long_short['buy_hold_return_pct']:.2f}%", flush=True)
        print(f"Number of Trades: {roi_long_short['num_trades']}", flush=True)
        
        # Compare strategies
        print(f"\n🎯 Strategy Comparison:", flush=True)
        improvement = roi_long_short['total_return_pct'] - roi_long_only['total_return_pct']
        print(f"Long-Short vs Long-Only: {improvement:+.2f}% difference", flush=True)
        
        # Plot both strategies for comparison
        print("\n📊 Generating comparison plots...", flush=True)
        
        # Plot Long-Only strategy
        plot_roi_analysis(roi_long_only, predictions_for_roi, actuals_for_roi, 
                         save_path=config['output']['plot_path'].replace('.png', '_roi_long_only.png'))
        
        # Plot Long-Short strategy  
        plot_roi_analysis(roi_long_short, predictions_for_roi, actuals_for_roi,
                         save_path=config['output']['plot_path'].replace('.png', '_roi_long_short.png'))
        
        print(f"✅ Plots saved:", flush=True)
        print(f"  Long-Only: {config['output']['plot_path'].replace('.png', '_roi_long_only.png')}", flush=True)
        print(f"  Long-Short: {config['output']['plot_path'].replace('.png', '_roi_long_short.png')}", flush=True)
        
    except Exception as e:
        print(f"ROI analysis failed: {e}", flush=True)
