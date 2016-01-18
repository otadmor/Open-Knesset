#!/usr/bin/python
# -*- coding: UTF-8 -*-

class_map = {
    u"אומנה" : u"אימוץ",
    u"אונס" : u"אלימות מינית",
    u"הטרדה מינית" : u"אלימות מינית",
    u"אחים שכולים" : u"קצבאות",
    u"אשראי" : u"בנקאות",
    u"אפליה" : u"שוויון",
    #u"בחירות לנשיאות",
    u"בחירות לכנסת" : u"בחירות",
    u"בית דין לעבודה" : u"ביטוח לאומי",
    # u"בית הדין לענייני משפחה",
    # u"בני זוג",
    # u"בנייה ירוקה",
    # u"בריאות הפה",
    u"בטחון" : u"ביטחון",
    u"בתי דין רבנים" : u"בית דין רבני",
    u"בתי דין רבניים" : u"בית דין רבני",
    # u"גז ונפט",
    u"גיור" : u"דת",
    u"דת ומדינה" : u"דת",
    # uחופש דת, יהדות, נישואים אזרחיים, נישואין וגירושין, דת ומדינה, גיור, דת 
    u"דיור מוגן" : u"אזרחים ותיקים",
    u"דיור ציבורי" : u"דיור בר השגה",
    # u"דיני חוזים",
    # uשוק חופשי, ריכוזיות, תחרותיות במשק
    u"הורים" : u"הורות",
    # u"הזכות להליך הוגן",
    # u"הימורים",
    # u"הכרזת מלחמה",
    # u"המגזר הבדואי",
    u"הסברה" : u"חינוך",
    # u"הסדר מדיני",
    u"הסתה לגזענות" : u"גזענות",
    u"הפליה" : u"שוויון",

    # u"השמה חוץ ביתית" : "קטינים ונוער", 
    u"השעייה" : u"שחיתות",
    # u"השתלת אברים",
    #u"התאגדות",
    u"התרת נישואים" : u"בית דין רבני",
    u"ועדת הכלכלה" : u"כלכלה",
    u"זיהום אוויר" : u"איכות הסביבה",
    u"פסולת" : u"איכות הסביבה",
    u"זיהום" : u"איכות הסביבה",
    u"חוק אוויר נקי" : u"איכות הסביבה",
    # u"זיכרון השואה",
    u"זכויות הנכים" : u"ניצולי שואה",
    u"זכויות חיילים" : u"מילואים",
    # u"זכויות יוצרים",
    u"חוק אוויר נקי" : u"איכות הסביבה",
    u"חוק גיל הפרישה" : u"פנסיה",
    u"חוק האזרחים הותיקים" : u"אזרחים ותיקים",
    u"חוק הביטוח הלאומי" : u"ביטוח לאומי",
    u"חוק הספורט" : u"ספורט",
    u"חוק הרשות השניה" : u"טלוויזיה",
    u"חוק התקשורת" : u"תקשורת",
    u"חוק חג המצות" : u"חגים ומועדים",
    u"חוק חינוך ממלכתי" : u"חינוך",
    u"חוק טל" : u"ישיבות הסדר",
    u"חוק לימוד חובה" : u"חינוך יסודי ועל יסודי",
    # u"חינוך בלתי-פורמלי",
    u"חללי צה\"ל" : u"צבא",
    u"חניה" : u"תחבורה",
    # u"חסכון",
    u"חריגות בניה" : u"תכנון ובניה",
    # u"טכנולוגיה",
    # u"טלוויזיה",
    # u"יבוא ויצוא",
    u"יהודה ושומרון" : u"יהודה ושומרון",
    u"ימי זכרון" : u"חגים ומועדים",
    # u"ישיבות",
    # u"יתמות",
    u"לסביות" : u"להט\"ב",
    u"לשון הרע" : u"חופש ביטוי",
    u"מבחן הכנסה" : u"נישואין וגירושין",
    u"מהגרים בלתי חוקיים" : u"מהגרי עבודה",
    u"מינהל מקרקעי ישראל" : u"מנהל מקרקעי ישראל",
    # u"משטר נשיאותי",
    u"משרד התחבורה, התשתיות הלאומיות והבטיחות בדרכים" : u"תחבורה",
    u"משרד התקשורת" : u"תקשורת",
    # u"נגב" : "פריפריה",
    u"נהיגה בשכרות" : u"בטיחות בדרכים",
    u"נהיגה תחת השפעת סמים" : u"בטיחות בדרכים",
    u"נוכחים-נפקדים" : u"הגירה וגבולות",
    # u"ניירות ערך",
    # u"נכי צה\"ל",
    # u"נפגעי פעולות איבה",
    # u"נשק",
    # u"סחר נשים",
    # u"סיוע לעסקים",
    # u"סכסוך עבודה",
    # u"סלולר",
    u"עובדי ציבור" : u"השירות הציבורי",
    # u"עיתונות",
    # u"פונדקאות",
    u"פינוי" : u"התנחלויות",
    u"פלסטינאים" : u"הסדר מדיני",
    u"פקודת מס הכנסה" : u"מס הכנסה",
    u"פרסומות" : u"פרסום ושיווק",
    # u"ציבור דתי",
    # u"צפון" : "פריפריה",
    # u"קהילה אתיופית",
    # u"קצבאות ילדים",
    # u"קרן קיימת לישראל",
    # u"ראש הממשלה",
    # u"שב\"כ",
    u"שביתה" : u"סכסוך עבודה",
    u"אי שוויון/שוויון" : u"שוויון",
    # u"שוק חופשי",
    u"שירות המדינה" : u"עובדי ציבור",
    # u"שירותי קהילה",
    u"שכר והוצאות שרים וחברי כנסת" : u"מנהל תקין",
    # u"שעון קיץ",
    u"שעת חירום" : u"תקנות שעת חירום",
    u"תושבי חוץ" : u"דיור",
    # u"תיווך מקרקעין",
    u"תכנון ובנייה" : u"תכנון ובניה",
    # u"תלונות הציבור",
    # u"תמלוגים",
    u"תנאי תעסוקה" : u"זכויות עובדים",
    # u"תנועות נוער" : "קטינים ונוער", 
    # u"תעודת זהות" : "משרד הפנים",
    # u"תעופה",
    # u"תרומות",
}

change_docs_tags = {
    11992 : u'תחבורה',
}