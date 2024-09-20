{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMd1ZR62mdKLoKNeve5okBe",
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
        "<a href=\"https://colab.research.google.com/github/Ahmadrafifaqiri/Ahmadrafifaqiri/blob/main/Hw1_Decision_Tree.run.sh\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install dataset"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hXWwNlEs85RI",
        "outputId": "96dceaf1-0735-4b94-85ca-62691541b3f1"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting dataset\n",
            "  Downloading dataset-1.6.2-py2.py3-none-any.whl.metadata (1.9 kB)\n",
            "Collecting sqlalchemy<2.0.0,>=1.3.2 (from dataset)\n",
            "  Downloading SQLAlchemy-1.4.54-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (10 kB)\n",
            "Collecting alembic>=0.6.2 (from dataset)\n",
            "  Downloading alembic-1.13.2-py3-none-any.whl.metadata (7.4 kB)\n",
            "Collecting banal>=1.0.1 (from dataset)\n",
            "  Downloading banal-1.0.6-py2.py3-none-any.whl.metadata (1.4 kB)\n",
            "Collecting Mako (from alembic>=0.6.2->dataset)\n",
            "  Downloading Mako-1.3.5-py3-none-any.whl.metadata (2.9 kB)\n",
            "Requirement already satisfied: typing-extensions>=4 in /usr/local/lib/python3.10/dist-packages (from alembic>=0.6.2->dataset) (4.12.2)\n",
            "Requirement already satisfied: greenlet!=0.4.17 in /usr/local/lib/python3.10/dist-packages (from sqlalchemy<2.0.0,>=1.3.2->dataset) (3.1.0)\n",
            "Requirement already satisfied: MarkupSafe>=0.9.2 in /usr/local/lib/python3.10/dist-packages (from Mako->alembic>=0.6.2->dataset) (2.1.5)\n",
            "Downloading dataset-1.6.2-py2.py3-none-any.whl (18 kB)\n",
            "Downloading alembic-1.13.2-py3-none-any.whl (232 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m233.0/233.0 kB\u001b[0m \u001b[31m8.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading banal-1.0.6-py2.py3-none-any.whl (6.1 kB)\n",
            "Downloading SQLAlchemy-1.4.54-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.6 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m1.6/1.6 MB\u001b[0m \u001b[31m48.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading Mako-1.3.5-py3-none-any.whl (78 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m78.6/78.6 kB\u001b[0m \u001b[31m6.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: banal, sqlalchemy, Mako, alembic, dataset\n",
            "  Attempting uninstall: sqlalchemy\n",
            "    Found existing installation: SQLAlchemy 2.0.35\n",
            "    Uninstalling SQLAlchemy-2.0.35:\n",
            "      Successfully uninstalled SQLAlchemy-2.0.35\n",
            "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
            "ipython-sql 0.5.0 requires sqlalchemy>=2.0, but you have sqlalchemy 1.4.54 which is incompatible.\u001b[0m\u001b[31m\n",
            "\u001b[0mSuccessfully installed Mako-1.3.5 alembic-1.13.2 banal-1.0.6 dataset-1.6.2 sqlalchemy-1.4.54\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_file = '/content/drive/My Drive/bank/train.csv'\n",
        "test_file = '/content/drive/My Drive/bank/test.csv'\n",
        "# Uncomment the next two lines if you want to use the car dataset instead\n",
        "# train_file = '/content/drive/My Drive/car/train.csv'\n",
        "# test_file = '/content/drive/My Drive/car/test.csv'"
      ],
      "metadata": {
        "id": "fwC5ohr_CTrd"
      },
      "execution_count": 23,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_train_path = \"/path/to/car/train.csv\"\n",
        "df_test_path = \"/path/to/car/test.csv\""
      ],
      "metadata": {
        "id": "PKmVmyKIDAwJ"
      },
      "execution_count": 34,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "print(\"Current working directory:\", os.getcwd())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VNxRK0VCCmkC",
        "outputId": "6054aa7c-04ee-4eeb-a7ba-be573cf88341"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Current working directory: /content\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "\n",
        "# List files in the 'car' directory\n",
        "if os.path.exists(\"car\"):\n",
        "    print(\"Files in 'car' directory:\", os.listdir(\"car\"))\n",
        "else:\n",
        "    print(\"'car' directory not found!\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "J_mKXy0GCont",
        "outputId": "7baa2f9f-9150-4ad1-b9a7-3ecaa9f25f5d"
      },
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "'car' directory not found!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "car_columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']\n",
        "bank_columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']\n"
      ],
      "metadata": {
        "id": "1eKBgZJY-S1I"
      },
      "execution_count": 37,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RrO_fMV__p7x",
        "outputId": "73fee57c-7850-4150-8201-c35b5379e21a"
      },
      "execution_count": 38,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_file = '/content/drive/My Drive/bank/train.csv'\n",
        "test_file = '/content/drive/My Drive/bank/test.csv'\n",
        "# Uncomment the next two lines if you want to use the car dataset instead\n",
        "# train_file = '/content/drive/My Drive/car/train.csv'\n",
        "# test_file = '/content/drive/My Drive/car/test.csv'"
      ],
      "metadata": {
        "id": "yYhiGUji-V8q"
      },
      "execution_count": 39,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "VhO_oxLr7qa_"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import pprint\n",
        "import dataset\n",
        "import numpy as np\n",
        "import statistics\n",
        "\n",
        "def calc_entropy(df):\n",
        "  attribute = df.keys()[-1]\n",
        "  values = df[attribute].unique()\n",
        "  entropy = 0.0\n",
        "  for value in values:\n",
        "    prob = df[attribute].value_counts()[value]/len(df[attribute])\n",
        "    entropy += -prob * np.log2(prob)\n",
        "  return np.float(entropy)\n",
        "\n",
        "def calc_majority_error(df):\n",
        "  attribute = df.keys()[-1]\n",
        "  values = df[attribute].unique()\n",
        "  majority_error = 1.0\n",
        "  for value in values:\n",
        "    prob = len(df[attribute][df[attribute] == value])/len(df[attribute])\n",
        "    majority_error = min(majority_error, prob)\n",
        "  return np.float(majority_error)\n",
        "\n",
        "def calc_gini_index(df):\n",
        "  attribute = df.keys()[-1]\n",
        "  values = df[attribute].unique()\n",
        "  gini_index = 1.0\n",
        "  for value in values:\n",
        "    prob = df[attribute].value_counts()[value]/len(df[attribute])\n",
        "    gini_index -= prob**2\n",
        "  return np.float(gini_index)\n",
        "\n",
        "def max_entropy_attribute(df, attribute):\n",
        "  target_attribute = df.keys()[-1]\n",
        "  target_values = df[target_attribute].unique()\n",
        "  attribute_values = df[attribute].unique()\n",
        "  avg_entropy = 0.0\n",
        "  for attrValue in attribute_values:\n",
        "    entropy = 0.0\n",
        "    for targetValue in target_values:\n",
        "      num = len(df[attribute][df[attribute] == attrValue][df[target_attribute] == targetValue])\n",
        "      den = len(df[attribute][df[attribute] == attrValue])\n",
        "      prob = num/den\n",
        "      entropy += -prob * np.log2(prob + 0.000001)\n",
        "    avg_entropy += (den/len(df))*entropy\n",
        "  return np.float(avg_entropy)\n",
        "\n",
        "def max_ME_attribute(df, attribute):\n",
        "  target_attribute = df.keys()[-1]\n",
        "  target_values = df[target_attribute].unique()\n",
        "  attribute_values = df[attribute].unique()\n",
        "  avg_ME = 0.0\n",
        "  for attrValue in attribute_values:\n",
        "    ME = 1.0\n",
        "    for targetValue in target_values:\n",
        "      num = len(df[attribute][df[attribute] == attrValue][df[target_attribute] == targetValue])\n",
        "      den = len(df[attribute][df[attribute] == attrValue])\n",
        "      prob = num/den\n",
        "      ME = min(ME, prob)\n",
        "    avg_ME += (den/len(df))*ME\n",
        "  return np.float(avg_ME)\n",
        "\n",
        "def max_GI_attribute(df, attribute):\n",
        "  target_attribute = df.keys()[-1]\n",
        "  target_values = df[target_attribute].unique()\n",
        "  attribute_values = df[attribute].unique()\n",
        "  avg_GI = 0.0\n",
        "  for attrValue in attribute_values:\n",
        "    GI = 1.0\n",
        "    for targetValue in target_values:\n",
        "      num = len(df[attribute][df[attribute] == attrValue][df[target_attribute] == targetValue])\n",
        "      den = len(df[attribute][df[attribute] == attrValue])\n",
        "      prob = num/den\n",
        "      GI -= prob**2\n",
        "    avg_GI += (den/len(df))*GI\n",
        "  return np.float(avg_GI)\n",
        "\n",
        "def checkAlreadyInSubTree(tree, candidate_attribute):\n",
        "  if candidate_attribute in tree.keys():\n",
        "    return True\n",
        "  return False\n",
        "\n",
        "def max_InfoGain_Attribute(df, algorithm, tree):\n",
        "  IG = []\n",
        "  if(algorithm == \"majority error\"):\n",
        "    for key in df.keys()[:-1]:\n",
        "      IG.append(calc_majority_error(df) - max_ME_attribute(df, key))\n",
        "    if(tree!=None and checkAlreadyInSubTree(tree, df.keys()[:-1][np.argmax(IG)])):\n",
        "      return df.keys()[:-1][np.argmax(IG[1:])]\n",
        "    return df.keys()[:-1][np.argmax(IG)]\n",
        "  elif(algorithm == \"entropy\"):\n",
        "    for key in df.keys()[:-1]:\n",
        "      IG.append(calc_entropy(df) - max_entropy_attribute(df, key))\n",
        "    if(tree!=None and checkAlreadyInSubTree(tree, df.keys()[:-1][np.argmax(IG)])):\n",
        "      return df.keys()[:-1][np.argmax(IG[1:])]\n",
        "    return df.keys()[:-1][np.argmax(IG)]\n",
        "  elif(algorithm == \"gini index\"):\n",
        "    for key in df.keys()[:-1]:\n",
        "      IG.append(calc_gini_index(df) - max_GI_attribute(df, key))\n",
        "    if(tree!=None and checkAlreadyInSubTree(tree, df.keys()[:-1][np.argmax(IG)])):\n",
        "      return df.keys()[:-1][np.argmax(IG[1:])]\n",
        "    return df.keys()[:-1][np.argmax(IG)]\n",
        "\n",
        "def get_subset(df, attribute, value):\n",
        "  return df[df[attribute] == value].reset_index(drop = True)\n",
        "\n",
        "def DecisionTreeClassifier(df, algorithm, maxDepth=100, tree = None):\n",
        "  selected_attribute = max_InfoGain_Attribute(df, algorithm, tree)\n",
        "  attribute_values = np.unique(df[selected_attribute])\n",
        "  target_attribute = df.keys()[-1]\n",
        "  if tree is None:\n",
        "    tree = {}\n",
        "    tree[selected_attribute] = {}\n",
        "  for attrValue in attribute_values:\n",
        "    subset = get_subset(df,selected_attribute,attrValue)\n",
        "    labels_for_attrValue, counts = np.unique(subset[target_attribute], return_counts = True)\n",
        "    if len(counts) == 1:\n",
        "      tree[selected_attribute][attrValue] = labels_for_attrValue[0]\n",
        "    else:\n",
        "      maxDepth-=1\n",
        "      if(maxDepth>0):\n",
        "        tree[selected_attribute][attrValue] = DecisionTreeClassifier(subset, algorithm, maxDepth)\n",
        "      else:\n",
        "        tree[selected_attribute][attrValue] = df[selected_attribute].mode()[0]\n",
        "  return tree\n",
        "\n",
        "def predict(instance, tree):\n",
        "  for node in tree.keys():\n",
        "    attribute = instance[node]\n",
        "    if(attribute not in tree[node].keys()):\n",
        "      return \"NotALeaf\"\n",
        "    value = tree[node][attribute]\n",
        "    if type(value) is dict:\n",
        "      return predict(instance, value)\n",
        "    else:\n",
        "      return value\n",
        "\n",
        "def evaluate(df_test, tree, train_or_test):\n",
        "  predicted_labels= []\n",
        "  correct_prediction = 0\n",
        "  for i in range(len(df_test)):\n",
        "    instance = df_test.iloc[i,:]\n",
        "    prediction = predict(instance, tree)\n",
        "    predicted_labels.append(prediction)\n",
        "  for i in range(len(df_test)):\n",
        "    if(df_test.iloc[i,-1] == predicted_labels[i]):\n",
        "      correct_prediction+=1\n",
        "  print(train_or_test + \" accuracy = \", correct_prediction/len(df_test), \"\\n\")\n",
        "\n",
        "#replace numeric attribute values with median threshold\n",
        "def median_thresholding(df, attribute):\n",
        "  threshold = df[attribute].median()\n",
        "  df[attribute] = (df[attribute] >= threshold).astype(int)\n",
        "\n",
        "#function to replace unknown values in training dataset with most common value\n",
        "def replace_unknown_values(df, attribute):\n",
        "  replacement = df[attribute][df[attribute]!=\"unknown\"].mode()[0]\n",
        "  df[attribute] = df[attribute].replace(to_replace = \"unknown\", value = replacement)\n",
        "\n",
        "def algorithm_option(option):\n",
        "  switcher = { 1 : \"majority error\", 2 : \"entropy\", 3 : \"gini index\"}\n",
        "  if(option not in [1,2,3]):\n",
        "    while(option not in [1, 2,3]):\n",
        "      print(\"Please select within provided options: \")\n",
        "      option = int(input())\n",
        "      if(option in [1,2,3]):\n",
        "        break\n",
        "  return switcher.get(option)\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "\n",
        "    choiceOfDataset = True if int(input(\"\\n1 - Car Dataset\\n2 - Bank Dataset\\nEnter your choice: \")) == 1 else False\n",
        "\n",
        "    if choiceOfDataset:\n",
        "      '''\n",
        "      get car dataset - maxDepth asked is 6\n",
        "      '''\n",
        "      maxDepth = int(input(\"\\nEnter maxdepth (1-6) or any number for full tree: \"))\n",
        "      df_train = pd.read_csv(\"car-4\\train.csv\")\n",
        "      df_train.columns = dataset.car_columns\n",
        "      df_test = pd.read_csv(\"car-4\\test.csv\")\n",
        "      df_test.columns = dataset.car_columns\n",
        "    else:\n",
        "      '''\n",
        "      get bank dataset - maxDepth asked in 16, numeric attributes are thresholded with median value,\n",
        "      attributes with unknown value are replaced with most common value\n",
        "      '''\n",
        "      maxDepth = int(input(\"\\nEnter maxdepth (1-16) or any number full tree: \"))\n",
        "      df_train = pd.read_csv(\"bank-4\\train.csv\")\n",
        "      df_train.columns = dataset.bank_columns\n",
        "      df_test = pd.read_csv(\"bank-4\\test.csv\")\n",
        "      df_test.columns = dataset.bank_columns\n",
        "      numeric_attributes = [\"age\", \"balance\", \"day\", \"duration\", \"campaign\", \"pdays\", \"previous\"]\n",
        "      for numeric_attr in numeric_attributes:\n",
        "        median_thresholding(df_train, numeric_attr)\n",
        "        median_thresholding(df_test, numeric_attr)\n",
        "      attributes_with_unknown = [\"job\", \"education\", \"contact\", \"poutcome\"]\n",
        "      for unknown_attrs in attributes_with_unknown:\n",
        "        replace_unknown_values(df_train, unknown_attrs)\n",
        "\n",
        "\n",
        "\n",
        "    if ((choiceOfDataset and maxDepth > 6) or (not(choiceOfDataset) and maxDepth>16)):\n",
        "      #run ID3 only once for maxDepth\n",
        "      option = int(input(\"\\nOptions:\\n1 - majority error\\n2 - entropy\\n3 - gini index\\nEnter your option: \"))\n",
        "      tree = DecisionTreeClassifier(df_train, algorithm_option(option))\n",
        "      evaluate(df_train, tree, \"training\")\n",
        "      evaluate(df_test, tree, \"testing\")\n",
        "    else:\n",
        "      #run ID3 for all depths till maxDepth\n",
        "      for i in range(1, maxDepth+1):\n",
        "        print(\"For a tree of maxdepth \", i ,\":\\n\")\n",
        "        tree = DecisionTreeClassifier(df_train, \"majority error\", i)\n",
        "        print(\"Using majority error,\\n\")\n",
        "        evaluate(df_train, tree, \"training\")\n",
        "        evaluate(df_test, tree, \"testing\")\n",
        "        tree = DecisionTreeClassifier(df_train, \"entropy\", i)\n",
        "        print(\"Using entropy,\\n\")\n",
        "        evaluate(df_train, tree, \"training\")\n",
        "        evaluate(df_test, tree, \"testing\")\n",
        "        tree = DecisionTreeClassifier(df_train, \"gini index\", i)\n",
        "        print(\"Using gini index,\\n\")\n",
        "        evaluate(df_train, tree, \"training\")\n",
        "        evaluate(df_test, tree, \"testing\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "_5zTNJEp-x6z"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}