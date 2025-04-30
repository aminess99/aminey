from flask import Flask, request, render_template, send_file
import yt_dlp
import uuid
import os
from threading import Timer

app = Flask(__name__)

# حذف الملف بعد الإرسال
def delete_file_later(path):
    def delete():
        try:
            os.remove(path)
        except:
            pass
    Timer(15, delete).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch', methods=['POST'])
def fetch():
    url = request.form['url']
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        # استخراج جميع الجودات المتاحة مع الحجم
        formats = []
        for f in info['formats']:
            if f.get('filesize') and f.get('vcodec') != 'none':  # فيديو فقط
                formats.append({
                    'format_id': f['format_id'],
                    'ext': f['ext'],
                    'resolution': f.get('resolution') or f.get('format_note'),
                    'filesize': round(f['filesize'] / 1024 / 1024, 2)
                })

        return render_template('download.html', title=info['title'], thumbnail=info['thumbnail'],
                               url=url, formats=formats)
    except Exception as e:
        return f"<p>حدث خطأ: {str(e)}</p>"

@app.route('/download/<format_id>', methods=['POST'])
def download(format_id):
    url = request.form['url']
    filename = f"{uuid.uuid4()}.%(ext)s"  # اسم فريد لكل فيديو

    # خيارات التحميل
    ydl_opts = {
        'format': format_id,
        'outtmpl': filename,
        'quiet': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            actual_file = ydl.prepare_filename(info)

        # حذف الملف بعد 15 ثانية من إرساله
        delete_file_later(actual_file)

        return send_file(actual_file, as_attachment=True)

    except Exception as e:
        return f"<p>حدث خطأ أثناء التحميل: {str(e)}</p>"

if __name__ == '__main__':
    app.run(debug=True)
