import matplotlib.pyplot as plt
import numpy as np

# Accuracy data (from your outputs)
depths = ['20', '56', '110']
plainnet_acc = [82.88, 84.36, 86.85]
resnet_acc = [84.27, 86.92, 88.57]

x = np.arange(len(depths))  # the label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, plainnet_acc, width, label='PlainNet', color='salmon')
bars2 = ax.bar(x + width/2, resnet_acc, width, label='ResNet', color='skyblue')

# Labels & Titles
ax.set_ylabel('Test Accuracy (%)')
ax.set_xlabel('Model Depth')
ax.set_title('PlainNet vs ResNet Accuracy on CIFAR-10')
ax.set_xticks(x)
ax.set_xticklabels(depths)
ax.set_ylim([80, 90])
ax.legend()

# Display bar values
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
# plt.show()
plt.savefig('plainnet_vs_resnet_accuracy.png')
