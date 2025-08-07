import os
from flask import Flask, request, render_template, send_from_directory, send_file, jsonify, url_for
from PIL import Image
import cv2
import numpy as np
import torch
from werkzeug.utils import secure_filename
import uuid
from threading import Timer

app = Flask(__name__)

# إعدادات المجلدات
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# إنشاء المجلدات إذا لم تكن موجودة
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs('weights', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 ميجابايت

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def delete_file_later(path, delay=300):  # 5 دقائق
    """حذف الملف بعد فترة معينة"""
    def delete():
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"خطأ في حذف الملف {path}: {e}")
    Timer(delay, delete).start()

class SimpleESRGAN:
    """نسخة مبسطة لتحسين الصور باستخدام OpenCV"""
    
    def __init__(self, scale=2):
        self.scale = scale
        
    def enhance_image(self, image_path, output_path):
        """تحسين الصورة باستخدام interpolation متقدم"""
        try:
            # قراءة الصورة
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("لا يمكن قراءة الصورة")
            
            # الحصول على أبعاد الصورة الأصلية
            height, width = img.shape[:2]
            
            # حساب الأبعاد الجديدة
            new_width = int(width * self.scale)
            new_height = int(height * self.scale)
            
            # تحسين الصورة باستخدام LANCZOS interpolation
            enhanced = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # تطبيق فلتر لتحسين الحدة
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # تحسين التباين والسطوع
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=10)
            
            # حفظ الصورة المحسنة
            cv2.imwrite(output_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return True
        except Exception as e:
            print(f"خطأ في تحسين الصورة: {e}")
            return False

# إنشاء مثيل من نموذج التحسين
enhancer = SimpleESRGAN(scale=2)

@app.route('/')
def index():
    return render_template('enhance.html')

@app.route('/enhance', methods=['POST'])
def enhance_image():
    if 'image' not in request.files:
        return jsonify({'error': 'لم يتم اختيار صورة'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400
    
    if file and allowed_file(file.filename):
        # إنشاء اسم ملف آمن وفريد
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        file_extension = filename.rsplit('.', 1)[1].lower()
        input_filename = f"{unique_id}_input.{file_extension}"
        output_filename = f"{unique_id}_enhanced.jpg"
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        try:
            # حفظ الصورة المرفوعة
            file.save(input_path)
            
            # تحسين الصورة
            success = enhancer.enhance_image(input_path, output_path)
            
            if success:
                # جدولة حذف الملفات بعد 5 دقائق
                delete_file_later(input_path)
                delete_file_later(output_path)
                
                return jsonify({
                    'success': True,
                    'original_image': url_for('uploaded_file', filename=input_filename),
                    'enhanced_image': url_for('output_file', filename=output_filename),
                    'message': 'تم تحسين الصورة بنجاح!'
                })
            else:
                return jsonify({'error': 'فشل في تحسين الصورة'}), 500
                
        except Exception as e:
            return jsonify({'error': f'خطأ في معالجة الصورة: {str(e)}'}), 500
    
    return jsonify({'error': 'نوع الملف غير مدعوم'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/status')
def health_check():
    """التحقق من حالة الخدمة"""
    return jsonify({
        'status': 'healthy',
        'service': 'image-enhancement-api',
        'gpu_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })

if __name__ == '__main__':
    print("🚀 بدء تشغيل خدمة تحسين جودة الصور...")
    print(f"📱 GPU متاح: {'نعم' if torch.cuda.is_available() else 'لا'}")
    
    # الحصول على البورت من متغير البيئة (Railway) أو استخدام البورت الافتراضي
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    
    print(f"🌐 التطبيق يعمل على: http://{host}:{port}")
    app.run(debug=False, host=host, port=port)
