# import schedule
# import time
# import subprocess

# def job():
#     print(f"Keeping alive at {time.ctime()}")
#     # Можно выполнить легкую команду
#     subprocess.run(["python", "-c", "print('alive')"])

# # Запланировать каждые 10 минут
# schedule.every(10).minutes.do(job)

# print("Keep-alive scheduler started. Press Ctrl+C to stop.")
# while True:
#     schedule.run_pending()
#     time.sleep(1)


import schedule
import time
import threading

def keep_alive_job():
    print(f"Keeping alive at {time.ctime()}")

def number_adder():
    """Функция, которая бесконечно складывает числа"""
    counter = 0
    sum_result = 0
    
    while True:
        sum_result += counter
        counter += 1
        
        # Выводим прогресс каждые 100000 итераций, чтобы не засорять вывод
        if counter % 100000 == 0:
            print(f"Сумма первых {counter} чисел: {sum_result}")
        if sum_result >= 100000000000:
            sum_result = 0
        
        # Небольшая пауза, чтобы не нагружать процессор
        time.sleep(0.001)

def main():
    # Запускаем планировщик для keep-alive сообщений
    # schedule.every(10).minutes.do(keep_alive_job)
    schedule.every(10).minutes.do(keep_alive_job)

    
    # Запускаем сложение чисел в отдельном потоке
    adder_thread = threading.Thread(target=number_adder, daemon=True)
    adder_thread.start()
    
    print("Keep-alive scheduler started. Number adder is running. Press Ctrl+C to stop.")
    
    # Основной цикл для обработки расписания
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()