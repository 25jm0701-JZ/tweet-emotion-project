
from Experiment1 import EmotionClassifier
from Experiment2 import roberta_EmotionClassifier
file_path = 'data/tweet_emotions.csv'   # 原版为 'tweet_emotions.csv'，改为从项目根目录运行

if __name__ == '__main__':
    classifier1 = EmotionClassifier(file_path)
    classifier1.run_Experiment1()

    classifier2 = roberta_EmotionClassifier(classifier1)
    classifier2.run_Experiment2()

