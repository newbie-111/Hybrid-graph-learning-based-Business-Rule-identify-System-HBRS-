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


#随机特征掩码和随机边缘丢弃
def random_feature_masking(nodes, mask_ratio=0.1):
    nodes = torch.FloatTensor(nodes)
    mask = torch.rand(nodes.shape) > mask_ratio
    return nodes * mask


def random_edge_dropping(edge_index, drop_ratio=0.1):
    mask = torch.rand(edge_index.shape[1]) > drop_ratio
    return edge_index[:, mask]


def contrastive_loss(z_i, z_j, temperature=0.5):
    """
    z_i and z_j are the representations of two augmentations of the same sample.
    temperature is a scaling factor for the logits.
    """
    batch_size = z_i.shape[0]

    # Normalize the representations
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    # Calculate similarity matrix
    similarity_matrix = torch.matmul(z_i, z_j.T) / temperature

    # Create labels
    labels = torch.arange(batch_size).long().to(z_i.device)

    # Calculate loss
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


class GCNWithContrastiveLearning(torch.nn.Module):
    def __init__(self, node_features, hidden_size, num_classes, dropout_rate=0.5):
        super(GCNWithContrastiveLearning, self).__init__()
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
        z = x  # Embeddings for contrastive learning
        x = F.softmax(self.fc(x), dim=1)
        return x, z


def train_model_with_contrastive_learning(train_nodes, train_labels, edge_index, model, optimizer, criterion, epochs, best_model_path, temperature=0.5):
    best_acc = 0
    loss_list = []
    accuracies = []

    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()

        # Original nodes and edges
        x = torch.FloatTensor(train_nodes)
        edge_index_original = edge_index

        # Augmented views for contrastive learning
        x_i = random_feature_masking(train_nodes, mask_ratio=0.1)
        edge_index_i = random_edge_dropping(edge_index, drop_ratio=0.1)

        x_j = random_feature_masking(train_nodes, mask_ratio=0.1)
        edge_index_j = random_edge_dropping(edge_index, drop_ratio=0.1)

        # Forward pass
        scores, embeddings = model(x, edge_index_original)
        _, z_i = model(x_i, edge_index_i)
        _, z_j = model(x_j, edge_index_j)

        target = torch.LongTensor(train_labels)
        loss_ce = criterion(scores, target)

        loss_contrastive = contrastive_loss(z_i, z_j, temperature)
        loss = loss_ce + loss_contrastive

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
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.show()

    plt.plot(accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Training Epochs")
    plt.show()


def test_model(test_nodes, test_labels, test_edge_index, model):
    x = torch.FloatTensor(test_nodes)

    with torch.no_grad():
        scores, _ = model(x, test_edge_index)  # Unpack the tuple to get scores

    predicted_labels = torch.argmax(scores, dim=1)
    true_labels = torch.LongTensor(test_labels)

    # Calculate precision, recall, and F1 score
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
    node_file = 'D:\\llvm-project\\testResult\\AST\\result_AutoPlan_interface2.csv'
    label_file = 'D:\\llvm-project\\testResult\\AST\\AutoPlan_interface.csv'
    edge_file = 'D:\\llvm-project\\testResult\\AST\\relation_AutoPlan_interface.csv'
    test_node_file = 'D:\\llvm-project\\testResult\\AST\\result_AutoPlan_task _test2.csv'
    test_label_file = 'D:\\llvm-project\\testResult\\AST\\AutoPlan_task _test.csv'
    test_edge_file = 'D:\\llvm-project\\testResult\\AST\\relation_AutoPlan_task _test1.csv'

    train_nodes, train_labels, edge_index, test_nodes, test_labels, test_edge_index = prepare_data(
        node_file, label_file, edge_file, test_node_file, test_label_file, test_edge_file)

    node_features = len(train_nodes[0])
    hidden_size = 32
    num_classes = len(set(train_labels))

    model = GCNWithContrastiveLearning(node_features=node_features, hidden_size=hidden_size, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train_model_with_contrastive_learning(train_nodes, train_labels, edge_index, model, optimizer, criterion, epochs=100, best_model_path="best_model.pth")

    predicted_labels, true_labels = test_model(test_nodes, test_labels, test_edge_index, model)

    results = pd.DataFrame({
        'True Labels': true_labels,
        'Predicted Labels': predicted_labels
    })

    results.to_csv('GCN_test_results.csv', index=False)


