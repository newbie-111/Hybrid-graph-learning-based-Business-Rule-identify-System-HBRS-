import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

# 从CSV文件中读取节点信息
def read_node_info(csv_file):
    df = pd.read_csv(csv_file, header=None)
    nodes = df.values.tolist()
    return nodes

# 从CSV文件中读取标签信息
def read_label_info(csv_file):
    df = pd.read_csv(csv_file)
    labels = df['label'].tolist()
    return labels

# 读取边信息
def read_edge_info(csv_file):
    data = pd.read_csv(csv_file)
    col_1 = data['source_nodes'].values.tolist()
    col_2 = data['target_nodes'].values.tolist()
    edge_index = torch.tensor([col_1, col_2], dtype=torch.long)
    print(f"Loaded edges with shape {edge_index.shape} from {csv_file}")
    return edge_index

# 准备训练数据和测试数据
def prepare_data(node_file, label_file, edge_file, test_node_file, test_label_file, test_edge_file):
    train_nodes = read_node_info(node_file)
    train_labels = read_label_info(label_file)
    edge_index = read_edge_info(edge_file)

    test_nodes = read_node_info(test_node_file)
    test_labels = read_label_info(test_label_file)
    test_edge_index = read_edge_info(test_edge_file)

    return train_nodes, train_labels, edge_index, test_nodes, test_labels, test_edge_index


#原理功能图
def plot_model_diagram(model):
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define the layers and their respective feature sizes
    layers = [
        {"name": "Input", "features": model.conv1.in_channels},
        {"name": "GCNConv1", "features": model.conv1.out_channels},
        {"name": "GCNConv2", "features": model.conv2.out_channels},
        {"name": "Output", "features": model.fc.out_features}
    ]

    # Define the positions for plotting
    y_positions = list(range(len(layers), 0, -1))

    for i, layer in enumerate(layers):
        ax.text(0.5, y_positions[i], f"{layer['name']} ({layer['features']} features)",
                ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(layers) + 1)
    ax.axis('off')
    ax.set_title("GCN Model Architecture")

    plt.show()


#准确率饼状图
def plot_accuracy_pie(true_labels, predicted_labels):
    # 计算正确预测的数量
    correct_predictions = sum([1 for t, p in zip(true_labels, predicted_labels) if t == p])
    total_predictions = len(true_labels)
    accuracy = correct_predictions / total_predictions

    # 绘制饼状图
    labels = 'Correct', 'Incorrect'
    sizes = [accuracy, 1 - accuracy]
    colors = ['#66b3ff','#ff9999']
    explode = (0.1, 0)  # 将 Correct 部分突出显示

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # 确保饼图是圆的
    plt.title('Accuracy Of GCN')
    plt.show()


class GCN(torch.nn.Module):
    def __init__(self, node_features, hidden_size, num_classes, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(node_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.softmax(self.fc(x), dim=1)
        return x


def train_model(train_nodes, train_labels, edge_index, model, optimizer, criterion, epochs, best_model_path):
    best_acc = 0
    loss_list = []
    accuracies = []
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()

        x = torch.FloatTensor(train_nodes)
        scores = model(x, edge_index)

        target = torch.LongTensor(train_labels)
        loss = criterion(scores, target)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

        accuracy = (torch.argmax(scores, dim=1) == target).float().mean().item()
        accuracies.append(accuracy)
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), best_model_path)

    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Loss")
    plt.show()
    # 绘制准确率折线图
    plt.plot(accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Training Epochs")
    plt.show()


def test_model(test_nodes, test_labels, test_edge_index, model):
    x = torch.FloatTensor(test_nodes)

    with torch.no_grad():
        scores = model(x, test_edge_index)

    predicted_labels = torch.argmax(scores, dim=1)
    true_labels = torch.LongTensor(test_labels)

    # 计算精确率、召回率和 F1 值
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=1)

    accuracy = (predicted_labels == true_labels).float().mean().item()
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print(predicted_labels)

    plot_accuracy_pie(true_labels, predicted_labels.tolist())
    plot_model_diagram(model)

    return predicted_labels.tolist(), true_labels.tolist()


# 主程序流程
if __name__ == "__main__":
    # 文件路径
    node_file = 'D:\\llvm-project\\testResult\\DFG\\result_AutoPlan_interface.csv'
    label_file = 'D:\\llvm-project\\testResult\\DFG\\AutoPlan_interface.csv'
    edge_file = 'D:\\llvm-project\\testResult\\DFG\\relation_AutoPlan_interface.csv'
    test_node_file = 'D:\\llvm-project\\testResult\\DFG\\result_AutoPlan_task_test.csv'
    test_label_file = 'D:\\llvm-project\\testResult\\DFG\\AutoPlan_task_test.csv'
    test_edge_file = 'D:\\llvm-project\\testResult\\DFG\\relation_AutoPlan_task_test.csv'

    # 准备数据
    train_nodes, train_labels, edge_index, test_nodes, test_labels, test_edge_index = prepare_data(
        node_file, label_file, edge_file, test_node_file, test_label_file, test_edge_file)

    # 模型参数
    node_features = len(train_nodes[0])  # 节点特征的维度
    hidden_size = 32
    num_classes = len(set(train_labels))  # 类别数，假设标签是从0开始的连续整数

    # 创建模型和优化器
    model = GCN(node_features=node_features, hidden_size=hidden_size, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练模型
    train_model(train_nodes, train_labels, edge_index, model, optimizer, criterion, epochs=100, best_model_path="best_model.pth")

    # 测试模型
    predicted_labels, true_labels = test_model(test_nodes, test_labels, test_edge_index, model)

    # 创建一个 DataFrame 并导出到 CSV 文件
    results = pd.DataFrame({
        'True Labels': true_labels,
        'Predicted Labels': predicted_labels
    })

    results.to_csv('GCN_test_results.csv', index=False)
