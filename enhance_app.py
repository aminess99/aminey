import os
from flask import Flask, request, render_template, send_from_directory, send_file, jsonify, url_for
from PIL import Image
import cv2
import numpy as np
import torch
from werkzeug.utils import secure_filename
import uuid
from threading import Timer
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

app = Flask(__name__)

# إعدادات المجلدات
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
TTS_OUTPUT_FOLDER = 'tts_outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}

# إنشاء المجلدات إذا لم تكن موجودة
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TTS_OUTPUT_FOLDER, exist_ok=True)
os.makedirs('weights', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['TTS_OUTPUT_FOLDER'] = TTS_OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 ميجابايت

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

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

# إنشاء مثيل من نموذج TTS (سيتم تحميله عند الحاجة)
tts_model = None

def get_tts_model():
    """تحميل نموذج TTS عند الحاجة"""
    global tts_model
    if tts_model is None:
        try:
            # استخدام CPU إذا لم يكن GPU متاحا
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🔄 محاولة تحميل نموذج TTS على {device}...")
            tts_model = ChatterboxTTS.from_pretrained(device=device)
            print(f"✅ تم تحميل نموذج TTS على {device}")
        except Exception as e:
            print(f"❌ خطأ في تحميل نموذج TTS: {e}")
            print("📝 ملاحظة: يتطلب تحميل النموذج اتصال إنترنت لتنزيل الملفات من HuggingFace")
            tts_model = "error"  # إشارة إلى وجود خطأ
    return tts_model if tts_model != "error" else None

@app.route('/')
def index():
    return render_template('unified.html')

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

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """تحويل النص إلى كلام"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'لم يتم تقديم النص'}), 400
    
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'النص فارغ'}), 400
    
    # التحقق من طول النص
    if len(text) > 1000:
        return jsonify({'error': 'النص طويل جداً (الحد الأقصى 1000 حرف)'}), 400
    
    try:
        # تحميل نموذج TTS
        model = get_tts_model()
        if model is None:
            return jsonify({'error': 'خدمة TTS غير متاحة حالياً - يتطلب اتصال إنترنت لتحميل النموذج', 'note': 'يرجى التأكد من اتصال الإنترنت والمحاولة مرة أخرى'}), 503
        
        # إنشاء ملف فريد
        unique_id = str(uuid.uuid4())
        output_filename = f"{unique_id}_tts.wav"
        output_path = os.path.join(app.config['TTS_OUTPUT_FOLDER'], output_filename)
        
        # تحويل النص إلى كلام
        wav = model.generate(text)
        
        # حفظ الملف الصوتي
        ta.save(output_path, wav, model.sr)
        
        # جدولة حذف الملف بعد 5 دقائق
        delete_file_later(output_path)
        
        return jsonify({
            'success': True,
            'audio_url': url_for('tts_file', filename=output_filename),
            'message': 'تم تحويل النص إلى كلام بنجاح!'
        })
        
    except Exception as e:
        print(f"خطأ في TTS: {e}")
        return jsonify({'error': f'خطأ في تحويل النص إلى كلام: {str(e)}'}), 500

@app.route('/tts_with_voice', methods=['POST'])
def text_to_speech_with_voice():
    """تحويل النص إلى كلام باستخدام صوت مرجعي"""
    if 'text' not in request.form:
        return jsonify({'error': 'لم يتم تقديم النص'}), 400
    
    text = request.form['text'].strip()
    if not text:
        return jsonify({'error': 'النص فارغ'}), 400
    
    if len(text) > 1000:
        return jsonify({'error': 'النص طويل جداً (الحد الأقصى 1000 حرف)'}), 400
    
    # التحقق من وجود ملف صوتي
    if 'audio' not in request.files:
        return jsonify({'error': 'لم يتم رفع ملف صوتي'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '' or not allowed_audio_file(audio_file.filename):
        return jsonify({'error': 'نوع الملف الصوتي غير مدعوم'}), 400
    
    try:
        # تحميل نموذج TTS
        model = get_tts_model()
        if model is None:
            return jsonify({'error': 'خدمة TTS غير متاحة حالياً - يتطلب اتصال إنترنت لتحميل النموذج', 'note': 'يرجى التأكد من اتصال الإنترنت والمحاولة مرة أخرى'}), 503
        
        # حفظ الملف الصوتي المرجعي
        unique_id = str(uuid.uuid4())
        audio_filename = secure_filename(audio_file.filename)
        audio_extension = audio_filename.rsplit('.', 1)[1].lower()
        reference_filename = f"{unique_id}_ref.{audio_extension}"
        reference_path = os.path.join(app.config['UPLOAD_FOLDER'], reference_filename)
        audio_file.save(reference_path)
        
        # إنشاء ملف الإخراج
        output_filename = f"{unique_id}_tts_voice.wav"
        output_path = os.path.join(app.config['TTS_OUTPUT_FOLDER'], output_filename)
        
        # تحويل النص إلى كلام باستخدام الصوت المرجعي
        wav = model.generate(text, audio_prompt_path=reference_path)
        
        # حفظ الملف الصوتي
        ta.save(output_path, wav, model.sr)
        
        # جدولة حذف الملفات بعد 5 دقائق
        delete_file_later(reference_path)
        delete_file_later(output_path)
        
        return jsonify({
            'success': True,
            'audio_url': url_for('tts_file', filename=output_filename),
            'message': 'تم تحويل النص إلى كلام بالصوت المرجعي بنجاح!'
        })
        
    except Exception as e:
        print(f"خطأ في TTS مع الصوت المرجعي: {e}")
        return jsonify({'error': f'خطأ في تحويل النص إلى كلام: {str(e)}'}), 500

@app.route('/tts_outputs/<filename>')
def tts_file(filename):
    """إرسال ملف TTS"""
    return send_from_directory(app.config['TTS_OUTPUT_FOLDER'], filename)
@app.route('/download/<filename>')
def download_file(filename):
    # البحث في المجلدين
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    
    if os.path.exists(output_path):
        return send_file(output_path, as_attachment=True)
    elif os.path.exists(upload_path):
        return send_file(upload_path, as_attachment=True)
    else:
        return jsonify({'error': 'الملف غير موجود'}), 404

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/status')
def health_check():
    """التحقق من حالة الخدمة"""
    # التحقق من حالة TTS
    tts_status = "unavailable"
    try:
        model = get_tts_model()
        if model is not None:
            tts_status = "available"
        else:
            tts_status = "requires_internet"
    except:
        tts_status = "error"
    
    return jsonify({
        'status': 'healthy',
        'services': {
            'image_enhancement': 'available',
            'text_to_speech': tts_status
        },
        'gpu_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'notes': {
            'tts': 'TTS service requires internet connection to download model on first use'
        }
    })

if __name__ == '__main__':
    print("🚀 بدء تشغيل خدمة تحسين جودة الصور...")
    print(f"📱 GPU متاح: {'نعم' if torch.cuda.is_available() else 'لا'}")
    
    # الحصول على البورت من متغير البيئة (Railway) أو استخدام البورت الافتراضي
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    
    print(f"🌐 التطبيق يعمل على: http://{host}:{port}")
    app.run(debug=False, host=host, port=port)
