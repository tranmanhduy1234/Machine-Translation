import shutil

# Kiểm tra ổ dữ liệu (Volume)
total, used, free = shutil.disk_usage("/workspace")
print(f"--- VOLUME (/workspace) ---")
print(f"Tổng: {total // (2**30)} GB")
print(f"Đã dùng: {used // (2**30)} GB")
print(f"Còn trống: {free // (2**30)} GB")

print("-" * 20)

# Kiểm tra ổ hệ thống (Container)
total_os, used_os, free_os = shutil.disk_usage("/")
print(f"--- SYSTEM (Container) ---")
print(f"Còn trống: {free_os // (2**30)} GB")