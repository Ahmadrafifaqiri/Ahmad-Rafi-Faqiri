{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMlK19ICFDbV7aIEFudZZ7f",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Ahmadrafifaqiri/Ahmadrafifaqiri/blob/main/Hw01_ML.run.sh\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Decision Tree [40 points + 10 bonus]"
      ],
      "metadata": {
        "id": "jxtXziMVK6wM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd"
      ],
      "metadata": {
        "id": "RxatJX18I8q-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data = pd.DataFrame({\n",
        "    'x1': [0, 0, 0, 1, 0, 1, 0],\n",
        "    'x2': [0, 1, 0, 0, 1, 1, 1],\n",
        "    'x3': [1, 0, 1, 0, 1, 0, 0],\n",
        "    'x4': [0, 0, 1, 1, 0, 0, 1],\n",
        "    'y': [0, 0, 1, 1, 0, 0, 0]\n",
        "})"
      ],
      "metadata": {
        "id": "jJsw96CDI-Ds"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def entropy(y):\n",
        "    probs = y.value_counts(normalize=True)\n",
        "    return -np.sum(probs * np.log2(probs))\n",
        "\n",
        "# Calculate information gain\n",
        "def information_gain(data, attribute):\n",
        "    # Calculate the entropy of the entire dataset\n",
        "    total_entropy = entropy(data['y'])\n",
        "\n",
        "\n",
        "    values = data[attribute].unique()\n",
        "    weighted_entropy = 0\n",
        "\n",
        "    for value in values:\n",
        "        subset = data[data[attribute] == value]\n",
        "        prob = len(subset) / len(data)\n",
        "        weighted_entropy += prob * entropy(subset['y'])\n",
        "\n",
        "    # Information gain is the reduction in entropy\n",
        "    return total_entropy - weighted_entropy"
      ],
      "metadata": {
        "id": "f2LPHvnFI-BH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "attributes = ['x1', 'x2', 'x3', 'x4']\n",
        "for attr in attributes:\n",
        "    gain = information_gain(data, attr)\n",
        "    print(f'Information Gain for {attr}: {gain:.4f}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yJqmHvbqI9-v",
        "outputId": "671a0389-ac51-449c-8b93-070e2861130b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Information Gain for x1: 0.0617\n",
            "Information Gain for x2: 0.4696\n",
            "Information Gain for x3: 0.0060\n",
            "Information Gain for x4: 0.4696\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data = {\n",
        "    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],\n",
        "    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Mild', 'Cool', 'Mild', 'Mild', 'Cool', 'Cool', 'Mild', 'Mild', 'Hot', 'Mild'],\n",
        "    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],\n",
        "    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong', 'Weak', 'Weak'],\n",
        "    'Play': ['No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']\n",
        "}\n",
        "\n",
        "df = pd.DataFrame(data)\n",
        "\n",
        "def majority_error(df, column, target):\n",
        "    \"\"\"\n",
        "    Calculate the Majority Error (ME) for a given attribute.\n",
        "    \"\"\"\n",
        "    groups = df.groupby(column)[target].value_counts(normalize=True).unstack(fill_value=0)\n",
        "    me = 1 - groups.max(axis=1)\n",
        "    weighted_me = (df.groupby(column).size() / len(df)).dot(me)\n",
        "    return weighted_me\n",
        "\n",
        "def gini_index(df, column, target):\n",
        "    \"\"\"\n",
        "    Calculate the Gini Index (GI) for a given attribute.\n",
        "    \"\"\"\n",
        "    groups = df.groupby(column)[target].value_counts(normalize=True).unstack(fill_value=0)\n",
        "    gini = 1 - (groups**2).sum(axis=1)\n",
        "    weighted_gini = (df.groupby(column).size() / len(df)).dot(gini)\n",
        "    return weighted_gini"
      ],
      "metadata": {
        "id": "0Ycd2jg-I97u"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Calculate Majority Error for each attribute:\n",
        "attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']\n",
        "me_results = {attr: majority_error(df, attr, 'Play') for attr in attributes}"
      ],
      "metadata": {
        "id": "zBH_ORe4I95i"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Calculate Gini Index for each attribute:\n",
        "gini_results = {attr: gini_index(df, attr, 'Play') for attr in attributes}\n",
        "\n",
        "print(\"Majority Error for each attribute:\")\n",
        "for attr, me in me_results.items():\n",
        "    print(f\"{attr}: {me:.3f}\")\n",
        "\n",
        "print(\"\\nGini Index for each attribute:\")\n",
        "for attr, gi in gini_results.items():\n",
        "    print(f\"{attr}: {gi:.3f}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LLDbWiYxI93D",
        "outputId": "f47472f3-f088-47dd-9b1a-59e295ceda48"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Majority Error for each attribute:\n",
            "Outlook: 0.286\n",
            "Temperature: 0.357\n",
            "Humidity: 0.286\n",
            "Wind: 0.357\n",
            "\n",
            "Gini Index for each attribute:\n",
            "Outlook: 0.343\n",
            "Temperature: 0.388\n",
            "Humidity: 0.367\n",
            "Wind: 0.405\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Identify the best attribute based on ME and GI:\n",
        "best_me_attr = min(me_results, key=me_results.get)\n",
        "best_gini_attr = min(gini_results, key=gini_results.get)\n",
        "\n",
        "print(f\"\\nBest attribute based on Majority Error: {best_me_attr}\")\n",
        "print(f\"Best attribute based on Gini Index: {best_gini_attr}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Xug8sfVCI90M",
        "outputId": "b236d106-f864-421d-da68-2616eb9f674f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Best attribute based on Majority Error: Outlook\n",
            "Best attribute based on Gini Index: Outlook\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Function to split dataset based on an attribute:\n",
        "def split_dataset(df, column, value):\n",
        "    \"\"\"\n",
        "    Split the dataset based on a given attribute and value.\n",
        "    \"\"\"\n",
        "    return df[df[column] == value]"
      ],
      "metadata": {
        "id": "KjdF1eHJI9sT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Split dataset based on the best attribute from Majority Error\n",
        "For recursive splitting,repeat the process for each subset"
      ],
      "metadata": {
        "id": "krlStYksJlFA"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "subset = split_dataset(df, best_me_attr, df[best_me_attr].unique()[0])\n",
        "print(f\"\\nSubset based on {best_me_attr} = {df[best_me_attr].unique()[0]}:\")\n",
        "print(subset)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_399NPM6JXW4",
        "outputId": "9ecd6846-f221-47e5-8964-3f6578b77070"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Subset based on Outlook = Sunny:\n",
            "   Outlook Temperature Humidity    Wind Play\n",
            "0    Sunny         Hot     High    Weak   No\n",
            "1    Sunny         Hot     High  Strong   No\n",
            "7    Sunny        Mild     High    Weak   No\n",
            "8    Sunny        Cool   Normal  Strong  Yes\n",
            "10   Sunny        Mild   Normal    Weak  Yes\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "Tx0FOM3XJjD6"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from collections import Counter\n",
        "from math import log2"
      ],
      "metadata": {
        "id": "ZH-VmsvUJXRX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data = {\n",
        "    'Outlook': ['Rain', 'Rain', 'Sunny', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Rain', 'Sunny', 'Rain', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Sunny'],\n",
        "    'Temperature': ['Hot', 'Hot', 'Mild', 'Hot', 'Hot', 'Mild', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Cool', 'Mild', 'Mild'],\n",
        "    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'Normal', 'High', 'High', 'Normal', 'Normal', 'High', 'High', 'High'],\n",
        "    'Wind': ['Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Strong', 'Weak'],\n",
        "    'Play': ['No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'No']\n",
        "}\n",
        "\n",
        "df = pd.DataFrame(data)\n",
        "\n",
        "new_instance = {'Outlook': None, 'Temperature': 'Mild', 'Humidity': 'Normal', 'Wind': 'Weak', 'Play': 'Yes'}\n",
        "most_common_outlook = df['Outlook'].mode()[0]\n",
        "df.loc[len(df)] = [most_common_outlook, new_instance['Temperature'], new_instance['Humidity'], new_instance['Wind'], new_instance['Play']]\n",
        "\n",
        "def entropy(attribute):\n",
        "    values, counts = np.unique(attribute, return_counts=True)\n",
        "    probs = counts / len(attribute)\n",
        "    return -np.sum(probs * np.log2(probs))\n",
        "\n",
        "\n",
        "def information_gain(df, feature, target):\n",
        "    total_entropy = entropy(df[target])\n",
        "    values, counts = np.unique(df[feature], return_counts=True)\n",
        "    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(df[df[feature] == values[i]][target]) for i in range(len(values))])\n",
        "    return total_entropy - weighted_entropy\n",
        "\n",
        "features = ['Outlook', 'Temperature', 'Humidity', 'Wind']\n",
        "target = 'Play'\n",
        "info_gains = {feature: information_gain(df, feature, target) for feature in features}\n",
        "\n",
        "print(\"Information Gain for each feature:\")\n",
        "for feature, gain in info_gains.items():\n",
        "    print(f\"{feature}: {gain:.3f}\")\n",
        "\n",
        "\n",
        "best_feature = max(info_gains, key=info_gains.get)\n",
        "print(f\"\\nThe best feature to split on is: {best_feature}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YX3LXRq_Jx_H",
        "outputId": "53b91273-e5c4-4ef6-dc81-459a3efe2556"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Information Gain for each feature:\n",
            "Outlook: 0.094\n",
            "Temperature: 0.004\n",
            "Humidity: 0.559\n",
            "Wind: 0.041\n",
            "\n",
            "The best feature to split on is: Humidity\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data = {\n",
        "    'Day': [1, 2, 7, 8, 9, 10, 11, 12, 14],\n",
        "    'Outlook': ['Sunny', 'Sunny', 'Sunny', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Sunny', 'Sunny'],\n",
        "    'Temperature': ['Hot', 'Hot', 'Mild', 'Mild', 'Mild', 'Mild', 'Mild', 'Mild', 'Mild'],\n",
        "    'Humidity': ['High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal'],\n",
        "    'Wind': ['Weak', 'Strong', 'Strong', 'Weak', 'Strong', 'Strong', 'Strong', 'Weak', 'Strong'],\n",
        "    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes']\n",
        "}\n",
        "\n",
        "df = pd.DataFrame(data)\n",
        "\n",
        "def entropy(y):\n",
        "    \"\"\"Calculate the entropy of labels.\"\"\"\n",
        "    proportions = y.value_counts(normalize=True)\n",
        "    return -np.sum(proportions * np.log2(proportions))\n",
        "\n",
        "def information_gain(df, feature, target):\n",
        "    \"\"\"Calculate the information gain of a feature.\"\"\"\n",
        "    # Calculate entropy of the target variable\n",
        "    entropy_before = entropy(df[target])\n",
        "\n",
        "    # Calculate weighted entropy after split\n",
        "    values = df[feature].unique()\n",
        "    entropy_after = 0\n",
        "    for value in values:\n",
        "        subset = df[df[feature] == value]\n",
        "        entropy_after += (len(subset) / len(df)) * entropy(subset[target])\n",
        "\n",
        "    # Information gain\n",
        "    return entropy_before - entropy_after\n",
        "\n",
        "# Calculate information gain for each feature\n",
        "features = ['Outlook', 'Temperature', 'Humidity', 'Wind']\n",
        "target = 'PlayTennis'\n",
        "\n",
        "info_gains = {}\n",
        "for feature in features:\n",
        "    info_gains[feature] = information_gain(df, feature, target)\n",
        "\n",
        "# Find the feature with the highest information gain\n",
        "best_feature = max(info_gains, key=info_gains.get)\n",
        "print(f\"The best feature to split on is: {best_feature} with an information gain of {info_gains[best_feature]}\")\n",
        "\n",
        "# Build the tree for the best feature\n",
        "def build_tree(df, feature, target):\n",
        "    \"\"\"Build a simple decision tree based on the best feature.\"\"\"\n",
        "    tree = {}\n",
        "    values = df[feature].unique()\n",
        "    for value in values:\n",
        "        subset = df[df[feature] == value]\n",
        "        if subset[target].nunique() == 1:\n",
        "            tree[value] = subset[target].iloc[0]\n",
        "        else:\n",
        "            # Recursively build the tree for each subset\n",
        "            subtree = {value: build_tree(subset, feature, target)}\n",
        "            tree.update(subtree)\n",
        "    return tree\n",
        "\n",
        "# Build and print the decision tree\n",
        "decision_tree = build_tree(df, best_feature, target)\n",
        "print(\"Decision Tree:\")\n",
        "print(decision_tree)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Nm9T6ZbeJXKA",
        "outputId": "2fc10efb-f2e5-4b02-8493-ec26892b6c9c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The best feature to split on is: Humidity with an information gain of 0.9182958340544896\n",
            "Decision Tree:\n",
            "{'High': 'No', 'Normal': 'Yes'}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PdlsESiLLWAq",
        "outputId": "1691b04a-475c-48ec-dc0f-eb9d6af9f4e1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from collections import Counter\n",
        "from sklearn.metrics import accuracy_score, confusion_matrix\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "train_file = '/content/drive/My Drive/bank/train.csv'\n",
        "test_file = '/content/drive/My Drive/bank/test.csv'\n",
        "train_file = '/content/drive/My Drive/car/train.csv'\n",
        "test_file = '/content/drive/My Drive/car/test.csv'\n"
      ],
      "metadata": {
        "id": "linR3hmFLZyw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_df = pd.read_csv(train_file)\n",
        "test_df = pd.read_csv(test_file)"
      ],
      "metadata": {
        "id": "tq6ZmOV5Lshz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "Display the first few rows of the data"
      ],
      "metadata": {
        "id": "tGLMjziBlDMN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "# Define file paths\n",
        "train_file = '/content/drive/My Drive/car/train.csv'\n",
        "test_file = '/content/drive/My Drive/car/test.csv'\n",
        "\n",
        "# Load the datasets using the specified structure\n",
        "train_df = pd.read_csv(train_file)\n",
        "test_df = pd.read_csv(test_file)\n",
        "\n",
        "# Display the first few rows of the training DataFrame\n",
        "train_df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 213
        },
        "id": "8JOHAp6KNegr",
        "outputId": "e111b3f4-c3c2-4071-b060-717217e433da"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "     low vhigh      4 4.1    big   med    acc\n",
              "0    low  high  5more   4    med  high  vgood\n",
              "1  vhigh   med      2   2    big  high  unacc\n",
              "2   high  high      2   2  small  high  unacc\n",
              "3  vhigh   low      3   2    big   low  unacc\n",
              "4   high  high      3   4    med   low  unacc"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-37bca1cf-c3a1-465c-a2cf-1df61759571e\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>low</th>\n",
              "      <th>vhigh</th>\n",
              "      <th>4</th>\n",
              "      <th>4.1</th>\n",
              "      <th>big</th>\n",
              "      <th>med</th>\n",
              "      <th>acc</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>low</td>\n",
              "      <td>high</td>\n",
              "      <td>5more</td>\n",
              "      <td>4</td>\n",
              "      <td>med</td>\n",
              "      <td>high</td>\n",
              "      <td>vgood</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>vhigh</td>\n",
              "      <td>med</td>\n",
              "      <td>2</td>\n",
              "      <td>2</td>\n",
              "      <td>big</td>\n",
              "      <td>high</td>\n",
              "      <td>unacc</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>high</td>\n",
              "      <td>high</td>\n",
              "      <td>2</td>\n",
              "      <td>2</td>\n",
              "      <td>small</td>\n",
              "      <td>high</td>\n",
              "      <td>unacc</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>vhigh</td>\n",
              "      <td>low</td>\n",
              "      <td>3</td>\n",
              "      <td>2</td>\n",
              "      <td>big</td>\n",
              "      <td>low</td>\n",
              "      <td>unacc</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>high</td>\n",
              "      <td>high</td>\n",
              "      <td>3</td>\n",
              "      <td>4</td>\n",
              "      <td>med</td>\n",
              "      <td>low</td>\n",
              "      <td>unacc</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-37bca1cf-c3a1-465c-a2cf-1df61759571e')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-37bca1cf-c3a1-465c-a2cf-1df61759571e button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-37bca1cf-c3a1-465c-a2cf-1df61759571e');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-6783841f-e972-4bea-b731-1b944edd86b5\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-6783841f-e972-4bea-b731-1b944edd86b5')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-6783841f-e972-4bea-b731-1b944edd86b5 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "train_df",
              "summary": "{\n  \"name\": \"train_df\",\n  \"rows\": 999,\n  \"fields\": [\n    {\n      \"column\": \"low\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 4,\n        \"samples\": [\n          \"vhigh\",\n          \"med\",\n          \"low\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"vhigh\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 4,\n        \"samples\": [\n          \"med\",\n          \"vhigh\",\n          \"high\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"4\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 4,\n        \"samples\": [\n          \"2\",\n          \"4\",\n          \"5more\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"4.1\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"4\",\n          \"2\",\n          \"more\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"big\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"med\",\n          \"big\",\n          \"small\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"med\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"high\",\n          \"low\",\n          \"med\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"acc\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 4,\n        \"samples\": [\n          \"unacc\",\n          \"good\",\n          \"vgood\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 44
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "test_df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 213
        },
        "id": "Uno0lHQsNmwl",
        "outputId": "d4433281-20a8-4141-8d56-15039cafbeaf"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   vhigh   high  5more  2  small  low  unacc\n",
              "0    low    low  5more  2  small  med  unacc\n",
              "1    low  vhigh      4  2    med  low  unacc\n",
              "2   high  vhigh      3  4    med  med  unacc\n",
              "3  vhigh    low      4  4    med  low  unacc\n",
              "4   high  vhigh  5more  4    med  low  unacc"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-0ddd0846-4b98-4143-9526-c6ebaf5a8761\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>vhigh</th>\n",
              "      <th>high</th>\n",
              "      <th>5more</th>\n",
              "      <th>2</th>\n",
              "      <th>small</th>\n",
              "      <th>low</th>\n",
              "      <th>unacc</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>low</td>\n",
              "      <td>low</td>\n",
              "      <td>5more</td>\n",
              "      <td>2</td>\n",
              "      <td>small</td>\n",
              "      <td>med</td>\n",
              "      <td>unacc</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>low</td>\n",
              "      <td>vhigh</td>\n",
              "      <td>4</td>\n",
              "      <td>2</td>\n",
              "      <td>med</td>\n",
              "      <td>low</td>\n",
              "      <td>unacc</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>high</td>\n",
              "      <td>vhigh</td>\n",
              "      <td>3</td>\n",
              "      <td>4</td>\n",
              "      <td>med</td>\n",
              "      <td>med</td>\n",
              "      <td>unacc</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>vhigh</td>\n",
              "      <td>low</td>\n",
              "      <td>4</td>\n",
              "      <td>4</td>\n",
              "      <td>med</td>\n",
              "      <td>low</td>\n",
              "      <td>unacc</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>high</td>\n",
              "      <td>vhigh</td>\n",
              "      <td>5more</td>\n",
              "      <td>4</td>\n",
              "      <td>med</td>\n",
              "      <td>low</td>\n",
              "      <td>unacc</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-0ddd0846-4b98-4143-9526-c6ebaf5a8761')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-0ddd0846-4b98-4143-9526-c6ebaf5a8761 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-0ddd0846-4b98-4143-9526-c6ebaf5a8761');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-51519112-3471-4ba3-9e89-6c49c408069d\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-51519112-3471-4ba3-9e89-6c49c408069d')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-51519112-3471-4ba3-9e89-6c49c408069d button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "test_df",
              "summary": "{\n  \"name\": \"test_df\",\n  \"rows\": 727,\n  \"fields\": [\n    {\n      \"column\": \"vhigh\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 4,\n        \"samples\": [\n          \"high\",\n          \"med\",\n          \"low\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"high\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 4,\n        \"samples\": [\n          \"vhigh\",\n          \"high\",\n          \"low\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"5more\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 4,\n        \"samples\": [\n          \"4\",\n          \"2\",\n          \"5more\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"2\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"2\",\n          \"4\",\n          \"more\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"small\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"small\",\n          \"med\",\n          \"big\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"low\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"med\",\n          \"low\",\n          \"high\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"unacc\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 4,\n        \"samples\": [\n          \"acc\",\n          \"good\",\n          \"unacc\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 45
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "car_train_file = '/content/drive/My Drive/car/train.csv'\n",
        "car_test_file = '/content/drive/My Drive/car/test.csv'\n",
        "bank_train_file = '/content/drive/My Drive/bank/train.csv'\n",
        "bank_test_file = '/content/drive/My Drive/bank/test.csv'"
      ],
      "metadata": {
        "id": "0OWfaeEONnnF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Check for missing values in car data\n",
        "print(car_train_df.isnull().sum())\n",
        "print(car_test_df.isnull().sum())\n",
        "\n",
        "# Optionally check for missing values in bank data\n",
        "print(bank_train_df.isnull().sum())\n",
        "print(bank_test_df.isnull().sum())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "siVWJYzb0rPD",
        "outputId": "d717819a-376f-4214-abb7-116e71c5afd0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "low      0\n",
            "vhigh    0\n",
            "4        0\n",
            "4.1      0\n",
            "big      0\n",
            "med      0\n",
            "acc      0\n",
            "dtype: int64\n",
            "vhigh    0\n",
            "high     0\n",
            "5more    0\n",
            "2        0\n",
            "small    0\n",
            "low      0\n",
            "unacc    0\n",
            "dtype: int64\n",
            "41           0\n",
            "services     0\n",
            "married      0\n",
            "secondary    0\n",
            "no           0\n",
            "0            0\n",
            "yes          0\n",
            "no.1         0\n",
            "unknown      0\n",
            "5            0\n",
            "may          0\n",
            "114          0\n",
            "2            0\n",
            "-1           0\n",
            "0.1          0\n",
            "unknown.1    0\n",
            "no.2         0\n",
            "dtype: int64\n",
            "41            0\n",
            "management    0\n",
            "single        0\n",
            "secondary     0\n",
            "no            0\n",
            "764           0\n",
            "no.1          0\n",
            "no.2          0\n",
            "cellular      0\n",
            "12            0\n",
            "jun           0\n",
            "230           0\n",
            "2             0\n",
            "-1            0\n",
            "0             0\n",
            "unknown       0\n",
            "no.3          0\n",
            "dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(car_train_df.columns)\n",
        "print(car_test_df.columns)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xS9o-8h72E7G",
        "outputId": "e3e6df79-68e9-4e19-b92c-d9a93bfea95f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Index(['low', 'vhigh', '4', '4.1', 'big', 'med', 'acc'], dtype='object')\n",
            "Index(['vhigh', 'high', '5more', '2', 'small', 'low', 'unacc'], dtype='object')\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "car_X_train = pd.get_dummies(car_train_df.drop('acc', axis=1))\n",
        "car_y_train = car_train_df['acc']\n",
        "\n",
        "car_X_test = pd.get_dummies(car_test_df.drop('unacc', axis=1))\n",
        "car_y_test = car_test_df['unacc']"
      ],
      "metadata": {
        "id": "Xo_6-fcC1RSw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "car_X_train, car_X_test = car_X_train.align(car_X_test, join='left', axis=1, fill_value=0)"
      ],
      "metadata": {
        "id": "fdr3Q1JS2TnN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Load the datasets\n",
        "car_train_df = pd.read_csv(car_train_file)\n",
        "car_test_df = pd.read_csv(car_test_file)\n",
        "bank_train_df = pd.read_csv(bank_train_file)\n",
        "bank_test_df = pd.read_csv(bank_test_file)"
      ],
      "metadata": {
        "id": "DdEalU1kP5pW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Define a function to calculate entropy\n",
        "def entropy(y):\n",
        "    proportions = y.value_counts(normalize=True)\n",
        "    return -np.sum(proportions * np.log2(proportions))\n",
        "\n",
        "# Define a function to calculate information gain\n",
        "def information_gain(X, y, feature):\n",
        "    original_entropy = entropy(y)\n",
        "    values = X[feature].unique()\n",
        "    weighted_entropy = 0\n",
        "    for value in values:\n",
        "        subset_y = y[X[feature] == value]\n",
        "        weighted_entropy += (len(subset_y) / len(y)) * entropy(subset_y)\n",
        "    return original_entropy - weighted_entropy\n",
        "\n",
        "# Define a function to calculate gini index\n",
        "def gini_index(y):\n",
        "    proportions = y.value_counts(normalize=True)\n",
        "    return 1 - np.sum(proportions ** 2)\n",
        "\n",
        "# Define a function to calculate gini gain\n",
        "def gini_gain(X, y, feature):\n",
        "    original_gini = gini_index(y)\n",
        "    values = X[feature].unique()\n",
        "    weighted_gini = 0\n",
        "    for value in values:\n",
        "        subset_y = y[X[feature] == value]\n",
        "        weighted_gini += (len(subset_y) / len(y)) * gini_index(subset_y)\n",
        "    return original_gini - weighted_gini\n",
        "\n",
        "# Define the ID3 algorithm with heuristics\n",
        "class DecisionTreeID3:\n",
        "    def __init__(self, max_depth=None, criterion='information_gain'):\n",
        "        self.max_depth = max_depth\n",
        "        self.criterion = criterion\n",
        "        self.tree = None\n",
        "\n",
        "    def fit(self, X, y):\n",
        "        self.tree = self._build_tree(X, y)\n",
        "\n",
        "    def _build_tree(self, X, y, depth=0):\n",
        "        if len(y.unique()) == 1:\n",
        "            return y.mode()[0]\n",
        "\n",
        "        if self.max_depth and depth >= self.max_depth:\n",
        "            return y.mode()[0]\n",
        "\n",
        "        if X.empty:\n",
        "            return y.mode()[0]\n",
        "\n",
        "        best_feature = self._choose_best_feature(X, y)\n",
        "        tree = {best_feature: {}}\n",
        "        for value in X[best_feature].unique():\n",
        "            subset_X = X[X[best_feature] == value].drop(best_feature, axis=1)\n",
        "            subset_y = y[X[best_feature] == value]\n",
        "            tree[best_feature][value] = self._build_tree(subset_X, subset_y, depth + 1)\n",
        "        return tree\n",
        "\n",
        "    def _choose_best_feature(self, X, y):\n",
        "        if self.criterion == 'information_gain':\n",
        "            return max(X.columns, key=lambda feature: information_gain(X, y, feature))\n",
        "        elif self.criterion == 'gini_index':\n",
        "            return max(X.columns, key=lambda feature: gini_gain(X, y, feature))\n",
        "        else:\n",
        "            raise ValueError(\"Unsupported criterion: {}\".format(self.criterion))\n",
        "\n",
        "    def predict(self, X):\n",
        "        return X.apply(lambda row: self._predict_single(row, self.tree), axis=1)\n",
        "\n",
        "    def _predict_single(self, row, tree):\n",
        "        if not isinstance(tree, dict):\n",
        "            return tree\n",
        "\n",
        "        feature = list(tree.keys())[0]\n",
        "        value = row[feature]\n",
        "        subtree = tree[feature].get(value, None)\n",
        "\n",
        "        if subtree is None:\n",
        "            return None\n",
        "\n",
        "        return self._predict_single(row, subtree)"
      ],
      "metadata": {
        "id": "Fovz03kUQBC1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# One-hot encode the training and test datasets\n",
        "car_X_train = pd.get_dummies(car_train_df.drop('acc', axis=1))\n",
        "car_y_train = car_train_df['acc']\n",
        "\n",
        "car_X_test = pd.get_dummies(car_test_df.drop('unacc', axis=1))\n",
        "car_y_test = car_test_df['unacc']\n",
        "\n",
        "# Ensure that the training and test sets have the same columns\n",
        "car_X_test = car_X_test.reindex(columns=car_X_train.columns, fill_value=0)\n",
        "\n",
        "# Define a function to evaluate the model\n",
        "def evaluate_model(model, X_train, y_train, X_test, y_test):\n",
        "    model.fit(X_train, y_train)\n",
        "    train_predictions = model.predict(X_train)\n",
        "    test_predictions = model.predict(X_test)\n",
        "    train_accuracy = accuracy_score(y_train, train_predictions)\n",
        "    test_accuracy = accuracy_score(y_test, test_predictions)\n",
        "    return train_accuracy, test_accuracy\n",
        "\n",
        "# Training and testing decision tree on car dataset\n",
        "depths = range(1, 7)\n",
        "criteria = ['information_gain', 'gini_index']\n",
        "\n",
        "results = []\n",
        "\n",
        "for criterion in criteria:\n",
        "    for depth in depths:\n",
        "        model = DecisionTreeID3(max_depth=depth, criterion=criterion)\n",
        "        train_accuracy, test_accuracy = evaluate_model(model, car_X_train, car_y_train, car_X_test, car_y_test)\n",
        "        results.append({'Depth': depth, 'Criterion': criterion, 'Train Accuracy': train_accuracy, 'Test Accuracy': test_accuracy})\n",
        "\n",
        "# Display results\n",
        "results_df = pd.DataFrame(results)\n",
        "print(results_df)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pPCNQ-5qQLIN",
        "outputId": "5435be59-55e0-4249-8873-3e4c1103a129"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "    Depth         Criterion  Train Accuracy  Test Accuracy\n",
            "0       1  information_gain        0.698699       0.702889\n",
            "1       2  information_gain        0.777778       0.222834\n",
            "2       3  information_gain        0.807808       0.374140\n",
            "3       4  information_gain        0.807808       0.257221\n",
            "4       5  information_gain        0.865866       0.257221\n",
            "5       6  information_gain        0.880881       0.257221\n",
            "6       1        gini_index        0.698699       0.702889\n",
            "7       2        gini_index        0.777778       0.222834\n",
            "8       3        gini_index        0.807808       0.374140\n",
            "9       4        gini_index        0.823824       0.331499\n",
            "10      5        gini_index        0.858859       0.257221\n",
            "11      6        gini_index        0.898899       0.299862\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def evaluate_model(model, X_train, y_train, X_test, y_test):\n",
        "    model.fit(X_train, y_train)\n",
        "    train_predictions = model.predict(X_train)\n",
        "    test_predictions = model.predict(X_test)\n",
        "\n",
        "    # Debugging print statements\n",
        "    print(\"Train predictions:\", train_predictions.unique())\n",
        "    print(\"Test predictions:\", test_predictions.unique())\n",
        "\n",
        "    train_accuracy = accuracy_score(y_train, train_predictions)\n",
        "    test_accuracy = accuracy_score(y_test, test_predictions)\n",
        "\n",
        "    return train_accuracy, test_accuracy"
      ],
      "metadata": {
        "id": "K5w52GOx6mRm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(bank_train_df.columns)\n",
        "print(bank_test_df.columns)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Mqzr5moy4U9C",
        "outputId": "79f6bef8-46b4-4867-ef06-d3dd05a1ed5e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Index(['41', 'services', 'married', 'secondary', 'no', '0', 'yes', 'no.1',\n",
            "       'unknown', '5', 'may', '114', '2', '-1', '0.1', 'unknown.1', 'no.2'],\n",
            "      dtype='object')\n",
            "Index(['41', 'management', 'single', 'secondary', 'no', '764', 'no.1', 'no.2',\n",
            "       'cellular', '12', 'jun', '230', '2', '-1', '0', 'unknown', 'no.3'],\n",
            "      dtype='object')\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(bank_train_df.head())\n",
        "print(bank_test_df.head())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KZs233cn4oyA",
        "outputId": "f59d1b1f-fb31-4637-a768-794fa728770b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   41     services   married  secondary  no     0  yes no.1   unknown   5  \\\n",
            "0  48  blue-collar    single  secondary  no   312  yes  yes  cellular   3   \n",
            "1  55   technician   married  secondary  no  1938   no  yes  cellular  18   \n",
            "2  54       admin.   married   tertiary  no    59  yes   no  cellular  10   \n",
            "3  34   management    single   tertiary  no  2646   no   no  cellular  14   \n",
            "4  49       admin.  divorced  secondary  no  1709  yes   no   unknown  12   \n",
            "\n",
            "   may  114  2   -1  0.1 unknown.1 no.2  \n",
            "0  feb  369  2   -1    0   unknown   no  \n",
            "1  aug  193  1  386    3   success  yes  \n",
            "2  jul  268  1   -1    0   unknown   no  \n",
            "3  apr  142  1   -1    0   unknown  yes  \n",
            "4  jun  106  1   -1    0   unknown   no  \n",
            "   41    management   single  secondary  no   764 no.1 no.2   cellular  12  \\\n",
            "0  39   blue-collar  married  secondary  no    49  yes   no   cellular  14   \n",
            "1  60       retired  married    primary  no     0   no   no  telephone  30   \n",
            "2  31  entrepreneur   single   tertiary  no   247  yes  yes    unknown   2   \n",
            "3  26       student   single    unknown  no  2020   no   no  telephone  28   \n",
            "4  58     housemaid  married    primary  no     0  yes   no  telephone   9   \n",
            "\n",
            "   jun  230  2   -1  0  unknown no.3  \n",
            "0  may  566  1  370  2  failure   no  \n",
            "1  jul  130  3   -1  0  unknown   no  \n",
            "2  jun  273  1   -1  0  unknown   no  \n",
            "3  jan   42  3   -1  0  unknown   no  \n",
            "4  jul  148  1   -1  0  unknown   no  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def _choose_best_feature(self, X, y):\n",
        "    if self.criterion == 'entropy':\n",
        "        return max(X.columns, key=lambda feature: entropy_gain(X, y, feature))\n",
        "    elif self.criterion == 'gini':\n",
        "        return max(X.columns, key=lambda feature: gini_gain(X, y, feature))\n",
        "    else:\n",
        "        raise ValueError(f\"Unsupported criterion: {self.criterion}\")"
      ],
      "metadata": {
        "id": "gZvxvOidonSB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(bank_train_df.columns)\n",
        "print(bank_test_df.columns)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZOsFnIoTpLrA",
        "outputId": "4eb06c3e-14f5-465e-d865-48ac3f3d9dc4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Index(['41', 'services', 'married', 'secondary', 'no', '0', 'yes', 'no.1',\n",
            "       'unknown', '5', 'may', '114', '2', '-1', '0.1', 'unknown.1', 'no.2'],\n",
            "      dtype='object')\n",
            "Index(['41', 'management', 'single', 'secondary', 'no', '764', 'no.1', 'no.2',\n",
            "       'cellular', '12', 'jun', '230', '2', '-1', '0', 'unknown', 'no.3'],\n",
            "      dtype='object')\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(bank_train_df.head())  # Print first few rows to see sample data\n",
        "print(bank_test_df.head())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "By7SGVbZpcH6",
        "outputId": "63891ec5-ebeb-436a-8f51-56279bccb2f6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   41     services   married  secondary  no     0  yes no.1   unknown   5  \\\n",
            "0  48  blue-collar    single  secondary  no   312  yes  yes  cellular   3   \n",
            "1  55   technician   married  secondary  no  1938   no  yes  cellular  18   \n",
            "2  54       admin.   married   tertiary  no    59  yes   no  cellular  10   \n",
            "3  34   management    single   tertiary  no  2646   no   no  cellular  14   \n",
            "4  49       admin.  divorced  secondary  no  1709  yes   no   unknown  12   \n",
            "\n",
            "   may  114  2   -1  0.1 unknown.1 no.2  \n",
            "0  feb  369  2   -1    0   unknown   no  \n",
            "1  aug  193  1  386    3   success  yes  \n",
            "2  jul  268  1   -1    0   unknown   no  \n",
            "3  apr  142  1   -1    0   unknown  yes  \n",
            "4  jun  106  1   -1    0   unknown   no  \n",
            "   41    management   single  secondary  no   764 no.1 no.2   cellular  12  \\\n",
            "0  39   blue-collar  married  secondary  no    49  yes   no   cellular  14   \n",
            "1  60       retired  married    primary  no     0   no   no  telephone  30   \n",
            "2  31  entrepreneur   single   tertiary  no   247  yes  yes    unknown   2   \n",
            "3  26       student   single    unknown  no  2020   no   no  telephone  28   \n",
            "4  58     housemaid  married    primary  no     0  yes   no  telephone   9   \n",
            "\n",
            "   jun  230  2   -1  0  unknown no.3  \n",
            "0  may  566  1  370  2  failure   no  \n",
            "1  jul  130  3   -1  0  unknown   no  \n",
            "2  jun  273  1   -1  0  unknown   no  \n",
            "3  jan   42  3   -1  0  unknown   no  \n",
            "4  jul  148  1   -1  0  unknown   no  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "from sklearn.metrics import accuracy_score\n",
        "from sklearn.tree import DecisionTreeClassifier  # Using scikit-learn's DecisionTreeClassifier\n",
        "\n",
        "# Print column names for verification\n",
        "print(\"Train Data Columns:\", bank_train_df.columns)\n",
        "print(\"Test Data Columns:\", bank_test_df.columns)\n",
        "\n",
        "# Define the target column based on your verification\n",
        "# Update this to the correct column name found in your datasets\n",
        "target_column = 'your_target_column'  # Replace with your actual target column name\n",
        "\n",
        "# Extract features and labels from the dataset\n",
        "if target_column in bank_train_df.columns and target_column in bank_test_df.columns:\n",
        "    bank_X_train = bank_train_df.drop(target_column, axis=1)  # Drop the target column to get the features\n",
        "    bank_y_train = bank_train_df[target_column]  # Extract the target column for labels\n",
        "    bank_X_test = bank_test_df.drop(target_column, axis=1)\n",
        "    bank_y_test = bank_test_df[target_column]\n",
        "\n",
        "    # Ensure there are no None values in y_train and y_test\n",
        "    bank_y_train = bank_y_train.replace({None: 'unknown'})\n",
        "    bank_y_test = bank_y_test.replace({None: 'unknown'})\n",
        "\n",
        "    # Function to evaluate the model\n",
        "    def evaluate_model(model, X_train, y_train, X_test, y_test):\n",
        "        model.fit(X_train, y_train)\n",
        "        train_predictions = model.predict(X_train)\n",
        "        test_predictions = model.predict(X_test)\n",
        "\n",
        "        # Clean None values from predictions and true values\n",
        "        train_predictions = pd.Series(train_predictions).replace({None: 'unknown'})\n",
        "        test_predictions = pd.Series(test_predictions).replace({None: 'unknown'})\n",
        "        y_train = y_train.replace({None: 'unknown'})\n",
        "        y_test = y_test.replace({None: 'unknown'})\n",
        "\n",
        "        # Debugging print statements\n",
        "        print(\"Train predictions:\", train_predictions.unique())\n",
        "        print(\"Test predictions:\", test_predictions.unique())\n",
        "\n",
        "        train_accuracy = accuracy_score(y_train, train_predictions)\n",
        "        test_accuracy = accuracy_score(y_test, test_predictions)\n",
        "\n",
        "        return train_accuracy, test_accuracy\n",
        "\n",
        "    # Initialize results list\n",
        "    results_bank = []\n",
        "\n",
        "    # Define criteria for the decision tree\n",
        "    criteria = ['gini', 'entropy']  # Ensure these criteria are supported by your model\n",
        "\n",
        "    for criterion in criteria:\n",
        "        for depth in range(1, 17):\n",
        "            # Create and train the model\n",
        "            model = DecisionTreeClassifier(max_depth=depth, criterion=criterion)\n",
        "            train_accuracy, test_accuracy = evaluate_model(model, bank_X_train, bank_y_train, bank_X_test, bank_y_test)\n",
        "            results_bank.append({'Depth': depth, 'Criterion': criterion, 'Train Accuracy': train_accuracy, 'Test Accuracy': test_accuracy})\n",
        "\n",
        "    # Display results\n",
        "    results_bank_df = pd.DataFrame(results_bank)\n",
        "    print(results_bank_df)\n",
        "else:\n",
        "    print(f\"Target column '{target_column}' not found in both train and test datasets.\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "heHDh7uOQSxr",
        "outputId": "51ae0b6d-9467-411f-d7aa-55645806549e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Train Data Columns: Index(['41', 'services', 'married', 'secondary', 'no', '0', 'yes', 'no.1',\n",
            "       'unknown', '5', 'may', '114', '2', '-1', '0.1', 'unknown.1', 'no.2'],\n",
            "      dtype='object')\n",
            "Test Data Columns: Index(['41', 'management', 'single', 'secondary', 'no', '764', 'no.1', 'no.2',\n",
            "       'cellular', '12', 'jun', '230', '2', '-1', '0', 'unknown', 'no.3'],\n",
            "      dtype='object')\n",
            "Target column 'your_target_column' not found in both train and test datasets.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(bank_test_df.columns)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nsQyQG2mm8o6",
        "outputId": "582ea9ce-bbd2-41aa-ab6c-7c3148c3c5a6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Index(['41', 'management', 'single', 'secondary', 'no', '764', 'no.1', 'no.2',\n",
            "       'cellular', '12', 'jun', '230', '2', '-1', '0', 'unknown', 'no.3'],\n",
            "      dtype='object')\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Print the column names of the training and test data\n",
        "print(\"Car Training Data Columns:\", car_train_df.columns)\n",
        "print(\"Car Test Data Columns:\", car_test_df.columns)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qbj1D4uxOrmk",
        "outputId": "efdaee8f-9f12-4648-eac2-4f8dcbd9ca33"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Car Training Data Columns: Index(['low', 'vhigh', '4', '4.1', 'big', 'med', 'acc'], dtype='object')\n",
            "Car Test Data Columns: Index(['vhigh', 'high', '5more', '2', 'small', 'low', 'unacc'], dtype='object')\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Load the datasets\n",
        "car_train_df = pd.read_csv(car_train_file)\n",
        "car_test_df = pd.read_csv(car_test_file)\n",
        "\n",
        "# Print column names to verify\n",
        "print(\"Car Training Data Columns:\", car_train_df.columns)\n",
        "print(\"Car Test Data Columns:\", car_test_df.columns)\n",
        "\n",
        "# Define a function to calculate entropy\n",
        "def entropy(y):\n",
        "    proportions = y.value_counts(normalize=True)\n",
        "    return -np.sum(proportions * np.log2(proportions))\n",
        "\n",
        "# Define a function to calculate information gain\n",
        "def information_gain(X, y, feature):\n",
        "    original_entropy = entropy(y)\n",
        "    values = X[feature].unique()\n",
        "    weighted_entropy = 0\n",
        "    for value in values:\n",
        "        subset_y = y[X[feature] == value]\n",
        "        weighted_entropy += (len(subset_y) / len(y)) * entropy(subset_y)\n",
        "    return original_entropy - weighted_entropy\n",
        "\n",
        "# Define a function to calculate gini index\n",
        "def gini_index(y):\n",
        "    proportions = y.value_counts(normalize=True)\n",
        "    return 1 - np.sum(proportions ** 2)\n",
        "\n",
        "# Define a function to calculate gini gain\n",
        "def gini_gain(X, y, feature):\n",
        "    original_gini = gini_index(y)\n",
        "    values = X[feature].unique()\n",
        "    weighted_gini = 0\n",
        "    for value in values:\n",
        "        subset_y = y[X[feature] == value]\n",
        "        weighted_gini += (len(subset_y) / len(y)) * gini_index(subset_y)\n",
        "    return original_gini - weighted_gini\n",
        "\n",
        "# Define the ID3 algorithm with heuristics\n",
        "class DecisionTreeID3:\n",
        "    def __init__(self, max_depth=None, criterion='information_gain'):\n",
        "        self.max_depth = max_depth\n",
        "        self.criterion = criterion\n",
        "        self.tree = None\n",
        "\n",
        "    def fit(self, X, y):\n",
        "        self.tree = self._build_tree(X, y)\n",
        "\n",
        "    def _build_tree(self, X, y, depth=0):\n",
        "        if len(y.unique()) == 1:\n",
        "            return y.mode()[0]\n",
        "\n",
        "        if self.max_depth and depth >= self.max_depth:\n",
        "            return y.mode()[0]\n",
        "\n",
        "        if X.empty:\n",
        "            return y.mode()[0]\n",
        "\n",
        "        best_feature = self._choose_best_feature(X, y)\n",
        "        tree = {best_feature: {}}\n",
        "        for value in X[best_feature].unique():\n",
        "            subset_X = X[X[best_feature] == value].drop(best_feature, axis=1)\n",
        "            subset_y = y[X[best_feature] == value]\n",
        "            tree[best_feature][value] = self._build_tree(subset_X, subset_y, depth + 1)\n",
        "        return tree"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "78qfPkNwQzRW",
        "outputId": "7006f881-a25c-4143-dcba-6b939945ea17"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Car Training Data Columns: Index(['low', 'vhigh', '4', '4.1', 'big', 'med', 'acc'], dtype='object')\n",
            "Car Test Data Columns: Index(['vhigh', 'high', '5more', '2', 'small', 'low', 'unacc'], dtype='object')\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "a. ID3 Algorithm with Categorical Attributes"
      ],
      "metadata": {
        "id": "U7VInPeENz9K"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "from sklearn.metrics import accuracy_score\n",
        "from typing import Any, Dict, Tuple, Union\n",
        "\n",
        "class DecisionTreeID3:\n",
        "    def __init__(self, max_depth: int = None, criterion: str = 'information_gain'):\n",
        "        self.max_depth = max_depth\n",
        "        self.criterion = criterion\n",
        "        self.tree = None\n",
        "\n",
        "    def fit(self, X: pd.DataFrame, y: pd.Series):\n",
        "        self.tree = self._fit(X, y, depth=0)\n",
        "\n",
        "    def _fit(self, X: pd.DataFrame, y: pd.Series, depth: int) -> Dict[str, Any]:\n",
        "        if len(set(y)) == 1:\n",
        "            return {'label': y.iloc[0]}\n",
        "\n",
        "        if self.max_depth is not None and depth >= self.max_depth:\n",
        "            return {'label': y.mode()[0]}\n",
        "\n",
        "        best_attribute = self._select_best_attribute(X, y)\n",
        "        tree = {best_attribute: {}}\n",
        "        for value in X[best_attribute].unique():\n",
        "            sub_X = X[X[best_attribute] == value].drop(columns=[best_attribute])\n",
        "            sub_y = y[X[best_attribute] == value]\n",
        "            tree[best_attribute][value] = self._fit(sub_X, sub_y, depth + 1)\n",
        "\n",
        "        return tree\n",
        "\n",
        "    def _select_best_attribute(self, X: pd.DataFrame, y: pd.Series) -> str:\n",
        "        if self.criterion == 'information_gain':\n",
        "            return self._best_information_gain(X, y)\n",
        "        elif self.criterion == 'gini_index':\n",
        "            return self._best_gini_index(X, y)\n",
        "        elif self.criterion == 'majority_error':\n",
        "            return self._best_majority_error(X, y)\n",
        "        else:\n",
        "            raise ValueError(\"Unknown criterion\")\n",
        "\n",
        "    def _best_information_gain(self, X: pd.DataFrame, y: pd.Series) -> str:\n",
        "        base_entropy = self._entropy(y)\n",
        "        best_gain = 0\n",
        "        best_attribute = None\n",
        "        for attribute in X.columns:\n",
        "            new_entropy = 0\n",
        "            for value in X[attribute].unique():\n",
        "                sub_y = y[X[attribute] == value]\n",
        "                new_entropy += (len(sub_y) / len(y)) * self._entropy(sub_y)\n",
        "            info_gain = base_entropy - new_entropy\n",
        "            if info_gain > best_gain:\n",
        "                best_gain = info_gain\n",
        "                best_attribute = attribute\n",
        "        return best_attribute\n",
        "\n",
        "    def _best_gini_index(self, X: pd.DataFrame, y: pd.Series) -> str:\n",
        "        best_gini = float('inf')\n",
        "        best_attribute = None\n",
        "        for attribute in X.columns:\n",
        "            gini = 0\n",
        "            for value in X[attribute].unique():\n",
        "                sub_y = y[X[attribute] == value]\n",
        "                prob = len(sub_y) / len(y)\n",
        "                gini += prob * (1 - sum([(sub_y.value_counts() / len(sub_y))**2]))\n",
        "            if gini < best_gini:\n",
        "                best_gini = gini\n",
        "                best_attribute = attribute\n",
        "        return best_attribute\n",
        "\n",
        "    def _best_majority_error(self, X: pd.DataFrame, y: pd.Series) -> str:\n",
        "        best_error = float('inf')\n",
        "        best_attribute = None\n",
        "        for attribute in X.columns:\n",
        "            error = 0\n",
        "            for value in X[attribute].unique():\n",
        "                sub_y = y[X[attribute] == value]\n",
        "                majority_class = sub_y.mode()[0]\n",
        "                error += (len(sub_y) / len(y)) * (1 - (sub_y == majority_class).mean())\n",
        "            if error < best_error:\n",
        "                best_error = error\n",
        "                best_attribute = attribute\n",
        "        return best_attribute\n",
        "\n",
        "    def _entropy(self, y: pd.Series) -> float:\n",
        "        probs = y.value_counts(normalize=True)\n",
        "        return -sum(probs * np.log2(probs + 1e-9))\n",
        "\n",
        "    def predict(self, X: pd.DataFrame) -> pd.Series:\n",
        "        return X.apply(self._predict_row, axis=1)\n",
        "\n",
        "    def _predict_row(self, row: pd.Series) -> Any:\n",
        "        tree = self.tree\n",
        "        while isinstance(tree, dict):\n",
        "            attribute = list(tree.keys())[0]\n",
        "            value = row[attribute]\n",
        "            tree = tree[attribute].get(value, {'label': 'unknown'})\n",
        "        return tree['label']\n"
      ],
      "metadata": {
        "id": "52Dtr5LoNWxD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "b. Evaluate Decision Trees for Car Datase"
      ],
      "metadata": {
        "id": "ialwVqC7NnfD"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from collections import Counter\n",
        "\n",
        "class DecisionTreeID3:\n",
        "    def __init__(self, max_depth=None):\n",
        "        self.max_depth = max_depth\n",
        "        self.tree = None\n",
        "\n",
        "    def fit(self, X, y):\n",
        "        self.tree = self._build_tree(X, y)\n",
        "\n",
        "    def _build_tree(self, X, y, depth=0):\n",
        "        if len(set(y)) == 1:  # All labels are the same\n",
        "            return y.iloc[0]\n",
        "\n",
        "        if self.max_depth is not None and depth >= self.max_depth:\n",
        "            return y.mode()[0]  # Return the most common label\n",
        "\n",
        "        best_feature = self._choose_best_feature(X, y)\n",
        "        tree = {best_feature: {}}\n",
        "\n",
        "        for value in X[best_feature].unique():\n",
        "            subset_X = X[X[best_feature] == value].drop(best_feature, axis=1)\n",
        "            subset_y = y[X[best_feature] == value]\n",
        "            tree[best_feature][value] = self._build_tree(subset_X, subset_y, depth + 1)\n",
        "\n",
        "        return tree\n",
        "\n",
        "    def _choose_best_feature(self, X, y):\n",
        "        return max(X.columns, key=lambda feature: self._information_gain(X, y, feature))\n",
        "\n",
        "    def _entropy(self, y):\n",
        "        counts = y.value_counts()\n",
        "        probabilities = counts / len(y)\n",
        "        return -np.sum(probabilities * np.log2(probabilities))\n",
        "\n",
        "    def _information_gain(self, X, y, feature):\n",
        "        original_entropy = self._entropy(y)\n",
        "        feature_entropy = 0\n",
        "\n",
        "        for value in X[feature].unique():\n",
        "            subset_y = y[X[feature] == value]\n",
        "            feature_entropy += (len(subset_y) / len(y)) * self._entropy(subset_y)\n",
        "\n",
        "        return original_entropy - feature_entropy\n",
        "\n",
        "    def predict(self, X):\n",
        "        return X.apply(self._predict_single, axis=1)\n",
        "\n",
        "    def _predict_single(self, row):\n",
        "        tree = self.tree\n",
        "        while isinstance(tree, dict):\n",
        "            feature = next(iter(tree))\n",
        "            feature_value = row[feature]\n",
        "            if feature_value not in tree[feature]:\n",
        "                return None  # Handle unseen values\n",
        "            tree = tree[feature][feature_value]\n",
        "        return tree\n",
        "\n",
        "# Example usage\n",
        "if __name__ == \"__main__\":\n",
        "    # Example data\n",
        "    data = {\n",
        "        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast'],\n",
        "        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot'],\n",
        "        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal'],\n",
        "        'Windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 'False', 'False', 'True', 'True', 'False'],\n",
        "        'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No']\n",
        "    }\n",
        "\n",
        "    df = pd.DataFrame(data)\n",
        "    X = df.drop('PlayTennis', axis=1)\n",
        "    y = df['PlayTennis']\n",
        "\n",
        "    model = DecisionTreeID3(max_depth=3)\n",
        "    model.fit(X, y)\n",
        "\n",
        "    print(\"Decision Tree:\")\n",
        "    print(model.tree)\n",
        "\n",
        "    test_data = pd.DataFrame({\n",
        "        'Outlook': ['Sunny', 'Rainy'],\n",
        "        'Temperature': ['Mild', 'Cool'],\n",
        "        'Humidity': ['High', 'Normal'],\n",
        "        'Windy': ['True', 'False']\n",
        "    })\n",
        "\n",
        "    predictions = model.predict(test_data)\n",
        "    print(\"Predictions:\")\n",
        "    print(predictions)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "COuFsYZfNXmy",
        "outputId": "3a9d53ad-5668-49b1-fecc-1c23a68fa72c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Decision Tree:\n",
            "{'Temperature': {'Hot': {'Outlook': {'Sunny': 'No', 'Overcast': {'Humidity': {'High': 'Yes', 'Normal': 'No'}}}}, 'Mild': {'Outlook': {'Rainy': 'Yes', 'Sunny': {'Humidity': {'High': 'No', 'Normal': 'Yes'}}, 'Overcast': 'Yes'}}, 'Cool': {'Outlook': {'Rainy': {'Windy': {'False': 'Yes', 'True': 'No'}}, 'Overcast': 'Yes', 'Sunny': 'Yes'}}}}\n",
            "Predictions:\n",
            "0     No\n",
            "1    Yes\n",
            "dtype: object\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#3. Modify for Numerical Attributes and Missing Values.\n",
        "\n",
        "a. Numerical Attributes"
      ],
      "metadata": {
        "id": "T6BUX1_iN3d2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from collections import Counter\n",
        "\n",
        "class DecisionTreeID3:\n",
        "    def __init__(self, max_depth=None):\n",
        "        self.max_depth = max_depth\n",
        "        self.tree = None\n",
        "\n",
        "    def fit(self, X, y):\n",
        "        self.tree = self._build_tree(X, y)\n",
        "\n",
        "    def _build_tree(self, X, y, depth=0):\n",
        "        if len(set(y)) == 1:  # All labels are the same\n",
        "            return y.iloc[0]\n",
        "\n",
        "        if self.max_depth is not None and depth >= self.max_depth:\n",
        "            return y.mode()[0]  # Return the most common label\n",
        "\n",
        "        best_feature = self._choose_best_feature(X, y)\n",
        "        tree = {best_feature: {}}\n",
        "\n",
        "        if isinstance(X[best_feature].dtype, np.number):\n",
        "            thresholds = self._find_thresholds(X[best_feature])\n",
        "            for threshold in thresholds:\n",
        "                subset_X_left = X[X[best_feature] <= threshold].drop(best_feature, axis=1)\n",
        "                subset_y_left = y[X[best_feature] <= threshold]\n",
        "                subset_X_right = X[X[best_feature] > threshold].drop(best_feature, axis=1)\n",
        "                subset_y_right = y[X[best_feature] > threshold]\n",
        "\n",
        "                tree[f'{best_feature} <= {threshold}'] = self._build_tree(subset_X_left, subset_y_left, depth + 1)\n",
        "                tree[f'{best_feature} > {threshold}'] = self._build_tree(subset_X_right, subset_y_right, depth + 1)\n",
        "        else:\n",
        "            for value in X[best_feature].unique():\n",
        "                subset_X = X[X[best_feature] == value].drop(best_feature, axis=1)\n",
        "                subset_y = y[X[best_feature] == value]\n",
        "                tree[value] = self._build_tree(subset_X, subset_y, depth + 1)\n",
        "\n",
        "        return tree\n",
        "\n",
        "    def _choose_best_feature(self, X, y):\n",
        "        features = X.columns\n",
        "        best_feature = None\n",
        "        best_info_gain = -float('inf')\n",
        "\n",
        "        for feature in features:\n",
        "            if isinstance(X[feature].dtype, np.number):\n",
        "                thresholds = self._find_thresholds(X[feature])\n",
        "                for threshold in thresholds:\n",
        "                    info_gain = self._information_gain(X, y, feature, threshold)\n",
        "                    if info_gain > best_info_gain:\n",
        "                        best_info_gain = info_gain\n",
        "                        best_feature = (feature, threshold)\n",
        "            else:\n",
        "                info_gain = self._information_gain(X, y, feature)\n",
        "                if info_gain > best_info_gain:\n",
        "                    best_info_gain = info_gain\n",
        "                    best_feature = feature\n",
        "\n",
        "        return best_feature\n",
        "\n",
        "    def _find_thresholds(self, feature):\n",
        "        \"\"\"Find unique thresholds for numerical feature.\"\"\"\n",
        "        thresholds = np.sort(feature.unique())\n",
        "        return (thresholds[:-1] + thresholds[1:]) / 2  # Midpoints between sorted unique values\n",
        "\n",
        "    def _entropy(self, y):\n",
        "        counts = y.value_counts()\n",
        "        probabilities = counts / len(y)\n",
        "        return -np.sum(probabilities * np.log2(probabilities))\n",
        "\n",
        "    def _information_gain(self, X, y, feature, threshold=None):\n",
        "        original_entropy = self._entropy(y)\n",
        "        feature_entropy = 0\n",
        "\n",
        "        if threshold is not None:\n",
        "            left_y = y[X[feature] <= threshold]\n",
        "            right_y = y[X[feature] > threshold]\n",
        "            feature_entropy = (len(left_y) / len(y)) * self._entropy(left_y) + (len(right_y) / len(y)) * self._entropy(right_y)\n",
        "        else:\n",
        "            for value in X[feature].unique():\n",
        "                subset_y = y[X[feature] == value]\n",
        "                feature_entropy += (len(subset_y) / len(y)) * self._entropy(subset_y)\n",
        "\n",
        "        return original_entropy - feature_entropy\n",
        "\n",
        "    def predict(self, X):\n",
        "        return X.apply(self._predict_single, axis=1)\n",
        "\n",
        "    def _predict_single(self, row):\n",
        "        tree = self.tree\n",
        "        while isinstance(tree, dict):\n",
        "            feature = next(iter(tree))\n",
        "            if isinstance(feature, tuple):\n",
        "                feature_name, threshold = feature\n",
        "                feature_value = row[feature_name]\n",
        "                if feature_value <= threshold:\n",
        "                    tree = tree[f'{feature_name} <= {threshold}']\n",
        "                else:\n",
        "                    tree = tree[f'{feature_name} > {threshold}']\n",
        "            else:\n",
        "                feature_value = row[feature]\n",
        "                if feature_value in tree:\n",
        "                    tree = tree[feature_value]\n",
        "                else:\n",
        "                    return None  # Handle unseen values\n",
        "        return tree\n",
        "\n",
        "# Example usage\n",
        "if __name__ == \"__main__\":\n",
        "    # Example data\n",
        "    data = {\n",
        "        'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],\n",
        "        'Income': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000],\n",
        "        'MaritalStatus': ['Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single', 'Married'],\n",
        "        'Buy': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']\n",
        "    }\n",
        "\n",
        "    df = pd.DataFrame(data)\n",
        "    X = df.drop('Buy', axis=1)\n",
        "    y = df['Buy']\n",
        "\n",
        "    model = DecisionTreeID3(max_depth=3)\n",
        "    model.fit(X, y)\n",
        "\n",
        "    print(\"Decision Tree:\")\n",
        "    print(model.tree)\n",
        "\n",
        "    test_data = pd.DataFrame({\n",
        "        'Age': [28, 55],\n",
        "        'Income': [65000, 105000],\n",
        "        'MaritalStatus': ['Single', 'Married']\n",
        "    })\n",
        "\n",
        "    predictions = model.predict(test_data)\n",
        "    print(\"Predictions:\")\n",
        "    print(predictions)"
      ],
      "metadata": {
        "id": "aGxpZsEPNkSp",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d711780f-8da5-47fd-f5bf-e93166f401f7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Decision Tree:\n",
            "{'Age': {}, 25: 'No', 30: 'Yes', 35: 'No', 40: 'Yes', 45: 'No', 50: 'Yes', 55: 'No', 60: 'Yes', 65: 'No', 70: 'Yes'}\n",
            "Predictions:\n",
            "0    None\n",
            "1      No\n",
            "dtype: object\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "b. Missing Values and Evaluation for Bank Dataset"
      ],
      "metadata": {
        "id": "5EEm4_I6OBcl"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Rename columns in bank_train_df\n",
        "bank_train_df.rename(columns={'services': 'management', '0': '764', '5': '12', '114': '230', 'unknown.1': 'unknown', 'no.2': 'no.3'}, inplace=True)\n",
        "\n",
        "# Rename columns in bank_test_df (if necessary)\n",
        "bank_test_df.rename(columns={'no.1': 'no', 'no.2': 'no.3'}, inplace=True)"
      ],
      "metadata": {
        "id": "eXjNoU4rro4a"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Drop columns not in the test set from the training set\n",
        "common_cols = bank_train_df.columns.intersection(bank_test_df.columns)\n",
        "bank_train_df = bank_train_df[common_cols]\n",
        "bank_test_df = bank_test_df[common_cols]"
      ],
      "metadata": {
        "id": "I8Dex_ccr9qe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "bank_train_df.columns = bank_train_df.columns.astype(str)\n",
        "bank_test_df.columns = bank_test_df.columns.astype(str)"
      ],
      "metadata": {
        "id": "VKC9tUsdsAGS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def handle_missing_values(df_train, df_test):\n",
        "    # Check for missing values\n",
        "    print(\"Missing values in training data:\")\n",
        "    print(df_train.isnull().sum())\n",
        "    print(\"Missing values in test data:\")\n",
        "    print(df_test.isnull().sum())\n",
        "\n",
        "    # Handle missing values\n",
        "    for column in df_train.columns:\n",
        "        if column in df_test.columns:\n",
        "            if df_train[column].dtype == 'object':\n",
        "                # Fill missing values with mode for categorical columns\n",
        "                mode = df_train[column].mode()[0]\n",
        "                df_train[column].fillna(mode, inplace=True)\n",
        "                df_test[column].fillna(mode, inplace=True)\n",
        "            else:\n",
        "                # Fill missing values with median for numerical columns\n",
        "                median = df_train[column].median()\n",
        "                df_train[column].fillna(median, inplace=True)\n",
        "                df_test[column].fillna(median, inplace=True)\n",
        "        else:\n",
        "            print(f\"Column {column} is not present in the test DataFrame.\")\n",
        "\n",
        "# Call the function to handle missing values\n",
        "handle_missing_values(bank_train_df, bank_test_df)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "3sxlVa8ssCH9",
        "outputId": "32132f2e-7719-46f2-ec41-3766373f576f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Missing values in training data:\n",
            "41            0\n",
            "management    0\n",
            "secondary     0\n",
            "no            0\n",
            "764           0\n",
            "unknown       0\n",
            "unknown       0\n",
            "12            0\n",
            "230           0\n",
            "2             0\n",
            "-1            0\n",
            "no.3          0\n",
            "dtype: int64\n",
            "Missing values in test data:\n",
            "41            0\n",
            "management    0\n",
            "secondary     0\n",
            "no            0\n",
            "no            0\n",
            "764           0\n",
            "unknown       0\n",
            "12            0\n",
            "230           0\n",
            "2             0\n",
            "-1            0\n",
            "no.3          0\n",
            "no.3          0\n",
            "dtype: int64\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-121-74c9b5131587>:20: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  df_test[column].fillna(median, inplace=True)\n",
            "<ipython-input-121-74c9b5131587>:15: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  df_test[column].fillna(mode, inplace=True)\n",
            "<ipython-input-121-74c9b5131587>:20: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  df_test[column].fillna(median, inplace=True)\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "AttributeError",
          "evalue": "'DataFrame' object has no attribute 'dtype'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-121-74c9b5131587>\u001b[0m in \u001b[0;36m<cell line: 25>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     23\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     24\u001b[0m \u001b[0;31m# Call the function to handle missing values\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 25\u001b[0;31m \u001b[0mhandle_missing_values\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbank_train_df\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbank_test_df\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m<ipython-input-121-74c9b5131587>\u001b[0m in \u001b[0;36mhandle_missing_values\u001b[0;34m(df_train, df_test)\u001b[0m\n\u001b[1;32m      9\u001b[0m     \u001b[0;32mfor\u001b[0m \u001b[0mcolumn\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mdf_train\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     10\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mcolumn\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mdf_test\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 11\u001b[0;31m             \u001b[0;32mif\u001b[0m \u001b[0mdf_train\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mcolumn\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdtype\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m'object'\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     12\u001b[0m                 \u001b[0;31m# Fill missing values with mode for categorical columns\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     13\u001b[0m                 \u001b[0mmode\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdf_train\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mcolumn\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmode\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pandas/core/generic.py\u001b[0m in \u001b[0;36m__getattr__\u001b[0;34m(self, name)\u001b[0m\n\u001b[1;32m   6202\u001b[0m         ):\n\u001b[1;32m   6203\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mname\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 6204\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mobject\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__getattribute__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mname\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   6205\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   6206\u001b[0m     \u001b[0;34m@\u001b[0m\u001b[0mfinal\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mAttributeError\u001b[0m: 'DataFrame' object has no attribute 'dtype'"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "bank_train_df.columns = bank_train_df.columns.str.strip()\n",
        "bank_test_df.columns = bank_test_df.columns.str.strip()"
      ],
      "metadata": {
        "id": "FHHmvPaGrtbB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Training Data Columns:\", car_train_df.columns)\n",
        "print(\"Test Data Columns:\", car_test_df.columns)\n"
      ],
      "metadata": {
        "id": "Z1xnl79ISMLE",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ae2fafe2-8c16-4c05-ac02-7b2e6c9a136f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training Data Columns: Index(['low', 'vhigh', '4', '4.1', 'big', 'med', 'acc'], dtype='object')\n",
            "Test Data Columns: Index(['vhigh', 'high', '5more', '2', 'small', 'low', 'unacc'], dtype='object')\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "car_train_df.columns = car_train_df.columns.str.strip()\n",
        "car_test_df.columns = car_test_df.columns.str.strip()"
      ],
      "metadata": {
        "id": "RNJVTaYCSMwv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Load car dataset with the correct paths\n",
        "car_train_df = pd.read_csv('/content/drive/My Drive/car/train.csv')\n",
        "car_test_df = pd.read_csv('/content/drive/My Drive/car/test.csv')\n",
        "\n",
        "# Debug column names\n",
        "print(\"Training Data Columns:\", car_train_df.columns)\n",
        "print(\"Test Data Columns:\", car_test_df.columns)\n",
        "\n",
        "# Strip any extra spaces from column names\n",
        "car_train_df.columns = car_train_df.columns.str.strip()\n",
        "car_test_df.columns = car_test_df.columns.str.strip()\n",
        "\n",
        "# Align the columns of the test dataset with the training dataset\n",
        "car_test_df = car_test_df.reindex(columns=car_train_df.columns, fill_value='missing')\n",
        "\n",
        "# Prepare data\n",
        "car_X_train = car_train_df.drop('acc', axis=1)\n",
        "car_y_train = car_train_df['acc']\n",
        "car_X_test = car_test_df.drop('acc', axis=1)\n",
        "car_y_test = car_test_df['acc']\n"
      ],
      "metadata": {
        "id": "xo5Iwq7WSa3W",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "058c98b1-ef1f-49b6-f748-84613e9584be"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training Data Columns: Index(['low', 'vhigh', '4', '4.1', 'big', 'med', 'acc'], dtype='object')\n",
            "Test Data Columns: Index(['vhigh', 'high', '5more', '2', 'small', 'low', 'unacc'], dtype='object')\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Identify columns in training set but not in test set\n",
        "missing_cols = [col for col in car_X_train.columns if col not in car_X_test.columns]\n",
        "print(\"Missing Columns in Test Set:\", missing_cols)"
      ],
      "metadata": {
        "id": "N6X1km0AW8re",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "7c205820-9450-4a09-80ed-7a96218227a5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Missing Columns in Test Set: []\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "etwN6UAtXCTS"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}