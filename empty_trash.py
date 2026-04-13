# pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

import pickle
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Если вы меняете этот scope, нужно будет удалить файл token.pickle
PATH = '/content/drive/MyDrive/miscellaneous/vscode_connection/gdrive_cleaning'
SCOPES = ['https://www.googleapis.com/auth/drive']

def get_authenticated_service():
    """
    Аутентификация пользователя и создание сервиса для работы с Google Drive API.
    Токен сохраняется в файл token.pickle для последующих запусков.
    """

    # creds - объект класса google.oauth2.credentials.Credentials, сначала учетных данных нет
    creds = None # bool(None) = False
    # Файл token.pickle хранит токены доступа пользователя
    if os.path.exists('token.pickle'):
        # with open(f'{PATH}/token.pickle', 'rb') as token:
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # Если нет валидных учетных данных, заставляем пользователя залогиниться
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # ВАЖНО: Замените 'credentials.json' на путь к вашему файлу с учетными данными OAuth 2.0
            # Скачайте его из Google Cloud Console.
            flow = InstalledAppFlow.from_client_secrets_file(
                # f'{PATH}/credentials.json', SCOPES)
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Сохраняем токен для следующего запуска
        # with open(f'{PATH}/token.pickle', 'wb') as token:
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    # Возвращаем сервис (клиентский объект) для работы с Google Drive API v3
    service = build('drive', 'v3', credentials=creds)
    return service

def empty_trash():
    """
    Основная функция: получает сервис и вызывает метод emptyTrash.
    """
    print("Начинаю процесс аутентификации...")
    service = get_authenticated_service()
    
    print("Очищаю корзину Google Диска...")
    try:
        # Вызов метода emptyTrash = полная очистка корзины
        # Согласно документации, при успешном выполнении возвращается пустой ответ (None) [citation:2][citation:3].
        service.files().emptyTrash().execute()
        print("✅ Корзина Google Диска успешно очищена.")
    except Exception as e:
        print(f"❌ Произошла ошибка при очистке корзины: {e}")

if __name__ == '__main__':
    empty_trash()