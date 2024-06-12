from email import message
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA


def eigenvalues_and_eigenvectors(matrix):
    # Перевірка, чи є матриця квадратною
    if matrix.shape[0] != matrix.shape[1]:#shape-показує розмірність матриці, наприклад 2 на 2,і тут перевіряється щоб матриця була квадратна
        raise ValueError("Матриця повинна бути квадратною")
    
    # Обчислення власних значень та власних векторів
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # Перевірка рівності A⋅v=λ⋅v для кожного власного значення та відповідного власного вектора
    for i in range(len(eigenvalues)):
        eigenvalue = eigenvalues[i]
        eigenvector = eigenvectors[:, i]
        left_side = np.dot(matrix, eigenvector)
        right_side = eigenvalue * eigenvector
        
        # Перевірка на близькість двох векторів
        if not  np.allclose(left_side, right_side):
            print(f"Перевірка провалилася для власного значення {eigenvalue} та власного вектора {eigenvector}")
        else:
            print(f"Перевірка успішна для власного значення {eigenvalue} та власного вектора {eigenvector}")
    
    return eigenvalues, eigenvectors

# Приклад використання функції
matrix = np.array([[4, -2], 
                   [1, 1]])
eigenvalues, eigenvectors = eigenvalues_and_eigenvectors(matrix)
print("Власні значення:", eigenvalues)
print("Власні вектори:\n", eigenvectors)

#зображення
image_raw = imread(r"C:\Users\Юлія\Downloads\tree.jpg")
plt.figure(figsize=[10,5])
plt.imshow(image_raw)
plt.title("Початкове кольорове зображення")
plt.axis('off')  
plt.show()
#Атрибут shape повертає кортеж, який містить кількість рядків, стовпців та каналів кольору (для кольорового зображення).
print(image_raw.shape)
# Ми розглядаємо третій вимір для каналу кольору,сумує значення вздовж третьої осі для кожного пікселя,
#  повертає 2вимірний масив (кортеж) де кожен елемент представляє суму значень трьох каналів для відповідного пікселя
image_sum = image_raw.sum(axis=2)
print(image_sum.shape)
plt.figure(figsize=[10, 5])
plt.imshow(image_sum, cmap=plt.cm.gray)
plt.title("Чорно-біле зображення")
plt.axis('off') 
plt.show()
# нормалізує значення пікселів у масиві image_sum, приводячи їх до діапазону від 0 до 1.
image_bw = image_sum/image_sum.max()
print(image_bw.max())
#еретворення двовимірного чорно-білого зображення у одновимірний масив, де кожен рядок представляє собою один піксель зображення.
image_bw_flat = image_bw.reshape(-1, image_bw.shape[1])#reshape-для зміни форми масиву shape -визначає кількість значень інтенсивності в кожному рядку

# Застосування PCA
pca = PCA()
pca.fit(image_bw_flat)

# Отримання кумулятивної дисперсії
var_cumu = np.cumsum(pca.explained_variance_ratio_) * 100

# Визначення кількості компонент для покриття 95% дисперсії
k = np.argmax(var_cumu > 95) + 1
print("Кількість компонент, необхідних для покриття 95% дисперсії: " + str(k))

# Побудова графіку кумулятивної дисперсії
plt.figure(figsize=[10,5])
plt.plot(var_cumu)
plt.title('Cumulative Explained Variance explained by the components')
plt.ylabel('Cumulative Explained variance (%)')
plt.xlabel('Principal components')
plt.axvline(x=k, color="k", linestyle="--", label=f'{k} components')
plt.axhline(y=95, color="r", linestyle="--", label='95% variance')
plt.legend()
plt.grid(True)
plt.show()

ipca = IncrementalPCA(n_components=k)#зменшення розмірності зображення до k
image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))#перетворене зображення, відновленим після зменшення розмірності до k головних компонент

plt.figure(figsize=[12,8])
plt.imshow(image_recon,cmap = plt.cm.gray)
plt.title("Реконструкція чорно-білого зображення")
plt.axis('off') 
plt.show()

def plot_at_k(k, subplot_idx):
    ipca = IncrementalPCA(n_components=k)
    image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))
 
    plt.subplot(2, 3, subplot_idx)
    plt.imshow(image_recon, cmap=plt.cm.gray)
    plt.title("Components: " + str(k))

# Задані кількості компонент для відновлення
ks = [5, 15, 25, 75, 100, 170]

plt.figure(figsize=[15, 9])

# Проходження по всім кількостям компонент і побудова графіків
for i, k in enumerate(ks):
    plot_at_k(k, i + 1)

plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.show()

def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message])
    eigenvalues, eigenvectors =np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)),np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)
    return encrypted_vector

def decrypt_message(encrypted_vector, key_matrix):
     eigenvalues, eigenvectors =np.linalg.eig(key_matrix)
     diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)),np.linalg.inv(eigenvectors))
     decrypted_vector = np.dot(np.linalg.inv(diagonalized_key_matrix), encrypted_vector)
     decrypted_message = ''.join([chr(int(np.round(num))) for num in decrypted_vector.real])
     return decrypted_message

message = "Hello, World!"
key_matrix = np.random.randint(0, 256, (len(message), len(message)))

encrypted_vector = encrypt_message(message, key_matrix)
print("Зашифрований вектор:", encrypted_vector)

decrypted_message = decrypt_message(encrypted_vector, key_matrix)
print("Розшифроване повідомлення:", decrypted_message)     