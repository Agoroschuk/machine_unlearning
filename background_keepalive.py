# pip install schedule
import schedule
import time
import subprocess

def job():
    print(f"Keeping alive at {time.ctime()}")
    # Можно выполнить легкую команду
    subprocess.run(["python", "-c", "print('alive')"])

# Запланировать каждые 10 минут
schedule.every(10).minutes.do(job)

print("Keep-alive scheduler started. Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    time.sleep(1)