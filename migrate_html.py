import os
import sys

app_file = r"c:\Users\yoqub\OneDrive\Desktop\AI 2\app.py"
tpl_dir = r"c:\Users\yoqub\OneDrive\Desktop\AI 2\templates"

try:
    with open(app_file, "r", encoding="utf-8") as f:
        content = f.read()

    # HTML blokini topish
    start_tag = 'HTML = """<!DOCTYPE html>'
    end_tag = '</html>"""'
    
    start_idx = content.find(start_tag)
    end_idx = content.find(end_tag)
    
    if start_idx == -1 or end_idx == -1:
        print("HTML bloki allaqachon ko'chirilgan yoki topilmadi.")
        sys.exit(0)
        
    # HTML kodni qirqib olish
    html_content = content[start_idx + len('HTML = """') : end_idx + len('</html>')]
    
    # templates papkasini yaratish va index.html ni saqlash
    os.makedirs(tpl_dir, exist_ok=True)
    with open(os.path.join(tpl_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
        
    # app.py dan HTML ni o'chirish va render_template_string ni almashtirish
    new_content = content[:start_idx] + content[end_idx + len(end_tag):]
    new_content = new_content.replace("from flask import render_template_string", "from flask import render_template")
    new_content = new_content.replace("render_template_string(\n        HTML,", "render_template(\n        'index.html',")
    
    with open(app_file, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print("✅ Muvaffaqiyatli! Barcha HTML kodlar templates/index.html ga ko'chirildi.")
    print("✅ app.py tozalandi.")
except Exception as e:
    print(f"Xatolik yuz berdi: {e}")
