from hatesonar import Sonar

def hate_speech_detection(texts):
    sonar = Sonar()
    phrases_by_line = " ".join(texts).split("  ")
    dynamic_black_list = []
    for sent in phrases_by_line:
        if sonar.ping(text=sent)['top_class'] != 'neither':
            [dynamic_black_list.append(i) for i in sent.split()]
    return dynamic_black_list