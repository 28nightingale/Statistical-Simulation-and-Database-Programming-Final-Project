vocabulary = []
K = 5 # Number of topics
num_of_words_per_topic = 200 # 200 * 5 =1000

topics = [
    'Movies_A',
    'Sports_B',
    'Finance_C',
    'Life_D',
    'Literature_E'
]

# generate 1000 words
for topic in topics:
    for i in range(1,num_of_words_per_topic+1):
        word = topic + str(i)
        vocabulary.append(word)

# Verfication step 
V_size = len(vocabulary)
print(f"Vocabulary size is: {V_size}")
print(f"Example words: {vocabulary[101]} and {vocabulary[202]}")

# Movies_A102 and Sports_83