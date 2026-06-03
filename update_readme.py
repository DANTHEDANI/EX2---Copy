import re

with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Function to increment header numbers
def bump_header(match):
    prefix = match.group(1)
    num = int(match.group(2))
    suffix = match.group(3)
    return f"{prefix}{num+1}{suffix}"

# Find the start of Section 1 to insert our new Section 1 before it
# We'll split the document into pre-section-1 and the rest
parts = re.split(r'(## 1\. מטרת הפרויקט)', content, maxsplit=1)

if len(parts) == 3:
    header = parts[0]
    rest = parts[1] + parts[2]
    
    # Bump all "## X." and "### X.Y"
    # Actually, we should only bump "## X." and "### X.Y"
    # Let's do it carefully.
    
    def bump_main(m):
        return f"## {int(m.group(1)) + 1}. {m.group(2)}"
    
    rest = re.sub(r'^## (\d+)\. (.*)$', bump_main, rest, flags=re.MULTILINE)
    
    def bump_sub(m):
        return f"### {int(m.group(1)) + 1}.{m.group(2)}"
        
    rest = re.sub(r'^### (\d+)\.(\d+.*)$', bump_sub, rest, flags=re.MULTILINE)

    # Now define the new Section 1
    new_section_1 = """## 1. קשר למצגת השיעור (Mapping to Course Presentation)

בהתאם לדרישות הפרויקט (חלק 2), להלן טבלה הממפה במפורש את המושגים שנלמדו בשיעור אל המימוש בפרויקט ואל המיקום המדויק שלהם בהמשך ה-README. 

| נושא מן השיעור | שקפים מומלצים להפניה | ביטוי נדרש בעבודה (ומיקום במסמך) |
|---|---|---|
| **מעבר מ-Q-Table ל-Function Approximation** | 3-6, 14-15 | **הסבר מדוע אי אפשר לנהל טבלת Q עבור חלונות זמן רציפים ורב-מאפיינים.**<br>ההסבר המלא מופיע ב[סעיף 14 - תשובות לשאלות למחשבה](#14-תשובות-לשאלות-למחשבה-סעיף-13-ממסמך-הדרישות). |
| **ניסוח בעיית RL** | 7-10 | **מיפוי מפורש של Agent, Environment, State, Action, Reward, Episode, Policy ו-Return.**<br>המיפוי המלא מופיע ב[סעיף 3 - מיפוי לבעיית RL](#3-מיפוי-לבעיית-rl-rl-mapping). |
| **נתונים כטנסור מצב** | 11-13 | **תיאור Pipeline הנתונים והצורה המדויקת של טנסור הקלט.**<br>מוסבר תחת [סעיף 4 - Dataset](#4-dataset-נתונים) וחלון הזמן מפורט ב[שאלה 2 בסעיף 14](#14-תשובות-לשאלות-למחשבה-סעיף-13-ממסמך-הדרישות). |
| **DQN ו-Dueling DQN** | 16-21 | **הסבר Value Stream, Advantage Stream וחישוב Q-values.**<br>מוסבר באריכות ב[סעיף 8.3 - Dueling Architecture](#83-dueling-architecture). |
| **ייצוב למידה Exploration** | 22-24 | **שימוש ב-epsilon-greedy, ב-Target Network ו-Experience Replay.**<br>מוסבר בסעיפים [8.4, 8.5 ו-8.6](#84-experience-replay-buffer-וייצוב-למידה). |
| **מחזור אימון מלא** | 25 | **תיאור reset, step, שמירת transitions, דגימת batch, עדכון Bellman target, loss ומשקלים.**<br>מתואר במפורש ב[סעיף 9 - תהליך אימון ו-Backtest](#9-תהליך-אימון-ו-backtest). |
| **Backtest וניתוח תוצאות** | 26-27 | **פירוש Equity Curve, Buy-and-Hold, Sharpe Ratio, Max Drawdown ו-Win Rate.**<br>מפורט תחת פסקת ה-Backtest ב[סעיף 9](#9-תהליך-אימון-ו-backtest). |
| **בדיקות וארכיטקטורה OOP** | 28-29 | **תרשימי מערכת, תרשים מחלקות, בדיקות ודיון באיכות הנדסית.**<br>מופיעים ב[סעיף 6 - ארכיטקטורת המערכת](#6-ארכיטקטורת-המערכת-וזרימת-נתונים-system-architecture), [סעיף 7 - תרשים מחלקות](#7-ארכיטקטורת-מחלקות-oop-architecture), וב[סעיף 11 - בדיקות ואיכות קוד](#11-בדיקות-ואיכות-קוד-tdd--tests). |
| **סיכום תאורטי** | 30-31 | **הדגשה שהסוכן לומד מדיניות החלטה ולא מנבא מחיר באופן ישיר.**<br>מודגש תחת [סעיף 2 - מטרת הפרויקט](#2-מטרת-הפרויקט-project-goal). |

---

"""
    
    final_content = header + new_section_1 + rest
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(final_content)
    print("README updated successfully.")
else:
    print("Could not find Section 1 marker.")
