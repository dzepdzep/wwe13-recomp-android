package com.toni.wwe13;

import android.app.Activity;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.hardware.input.InputManager;
import android.os.Handler;
import android.os.Looper;
import android.view.InputDevice;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;

/**
 * On-screen touch controls laid out for WWE '13 (not a generic Xbox pad): every button is named after what it does in
 * a match. They drive a virtual gamepad in libmain (touch_pad.cpp) that joins player 1 next to any real controller.
 *
 * Left: movement stick (push it to the edge to run), RUN (tap to switch running on or off) and CLIMB / TAG. Right: STRIKE, GRAPPLE, WHIP / PIN,
 * FINISHER, REVERSAL, LIMB / ITEM and a small right stick (reposition the opponent; tap = store a finisher).
 * Top: taunt d-pad, BACK and PAUSE. The game's own menus work with the stick, GRAPPLE (= A) and WHIP / PIN (= B).
 *
 * Shown according to the launcher's "Touch controls" setting: automatic (hidden while a controller is connected),
 * always or never. The overlay is a panel window, because the game surface is placed on top of the activity window.
 * The ✎ button (top right) opens the layout editor: drag any button, tap one and use − / + to resize it; the layout is
 * saved in the launcher's preferences ("touch_layout").
 */
final class TouchControls extends View implements InputManager.InputDeviceListener {
    // SDL_GamepadButton / SDL_GamepadAxis numbers.
    private static final int A = 0, B = 1, X = 2, Y = 3, BACK = 4, START = 6, LS = 7, RS = 8, LB = 9, RB = 10,
            UP = 11, DOWN = 12, LEFT = 13, RIGHT = 14;
    private static final int AX_LX = 0, AX_LY = 1, AX_RX = 2, AX_RY = 3, AX_LT = 4, AX_RT = 5;
    private static final int TRIGGER = 100;  // button ids >= TRIGGER are trigger axes (id - TRIGGER)

    static native boolean nativeAttach();
    static native void nativeDetach();
    static native void nativeButton(int button, boolean down);
    static native void nativeAxis(int axis, float value);

    /** A round button: position and radius in fractions of the screen height (x from the left or right edge). */
    private static final class Btn {
        final String label, sub;
        final int id, color;
        final float dfx, dfy, dfr;  // defaults
        float fx, fy, fr;
        final boolean fromRight;
        float cx, cy, r;
        int pointer = -1;
        long downTime;
        Btn(String label, String sub, int id, int color, boolean fromRight, float fx, float fy, float fr) {
            this.label = label; this.sub = sub; this.id = id; this.color = color;
            this.fromRight = fromRight; this.fx = dfx = fx; this.fy = dfy = fy; this.fr = dfr = fr;
        }
    }

    private static final int GREEN = 0xFF2E9E4F, RED = 0xFFC62828, BLUE = 0xFF1E5FB8, GOLD = 0xFFD4A017,
            GREY = 0xFF5A5A5A, ORANGE = 0xFFE0662B, PURPLE = 0xFF7B3FA8;

    private final Btn[] buttons = {
            // Right hand: the four actions around the thumb, REVERSAL above them.
            new Btn("GRAPPLE", "A", A, GREEN, true, 0.30f, 0.84f, 0.115f),
            new Btn("STRIKE", "X", X, BLUE, true, 0.52f, 0.66f, 0.115f),
            new Btn("WHIP", "PIN · B", B, RED, true, 0.11f, 0.62f, 0.105f),
            new Btn("FINISHER", "SIG · Y", Y, GOLD, true, 0.31f, 0.46f, 0.10f),
            new Btn("REVERSAL", "RT", TRIGGER + AX_RT, PURPLE, true, 0.13f, 0.30f, 0.125f),
            new Btn("LIMB", "ITEM · RB", RB, GREY, true, 0.50f, 0.36f, 0.075f),
            // Left hand, next to the stick.
            new Btn("RUN", "TAP: ON / OFF", TRIGGER + AX_LT, ORANGE, false, 0.62f, 0.86f, 0.085f),
            new Btn("CLIMB", "TAG · LB", LB, GREY, false, 0.66f, 0.60f, 0.075f),
            // Top row.
            new Btn("PAUSE", "", START, GREY, false, 0.95f, 0.07f, 0.05f),
            new Btn("BACK", "", BACK, GREY, false, 0.80f, 0.07f, 0.05f),
            new Btn("▲", "TAUNT", UP, GREY, false, 0.17f, 0.10f, 0.05f),
            new Btn("◀", "", LEFT, GREY, false, 0.07f, 0.20f, 0.05f),
            new Btn("▶", "", RIGHT, GREY, false, 0.27f, 0.20f, 0.05f),
            new Btn("▼", "", DOWN, GREY, false, 0.17f, 0.30f, 0.05f),
            // The right stick pad (drawn and handled separately; listed here so the editor can move and resize it).
            new Btn("", "", RS_PAD, GREY, true, 0.74f, 0.86f, 0.075f),
            new Btn("✎", "", EDIT, GREY, true, 0.06f, 0.07f, 0.04f),
    };
    private static final int RS_PAD = 200, EDIT = 201;  // not gamepad buttons
    private Btn rsBtn() { return buttons[buttons.length - 2]; }

    // Movement stick: appears where the left thumb lands (left half, lower part). Right stick: fixed and small.
    private int stickPointer = -1;
    private float stickBaseX, stickBaseY, stickX, stickY, stickR;
    private boolean running;     // stick pushed to the edge
    private boolean runToggled;  // RUN tapped on
    private boolean ltDown;      // what the game currently gets for LT
    private boolean vibrate;
    // Layout editor.
    private boolean editing;
    private Btn editSel, editDrag;
    private int editPointer = -1;
    private float editOffX, editOffY;
    private final RectF[] editBar = new RectF[4];  // −, +, RESET, DONE
    private int rsPointer = -1;
    private float rsCx, rsCy, rsR, rsX, rsY;
    private long rsDownTime;

    private final Paint fill = new Paint(Paint.ANTI_ALIAS_FLAG), ring = new Paint(Paint.ANTI_ALIAS_FLAG),
            text = new Paint(Paint.ANTI_ALIAS_FLAG), small = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final float alpha;
    private final int mode;  // 0 automatic, 1 always, 2 off
    private final Activity activity;
    private final android.content.SharedPreferences prefs;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private boolean attached, added, shown;

    static TouchControls install(Activity activity) {
        android.content.SharedPreferences p = activity.getSharedPreferences("launcher", Context.MODE_PRIVATE);
        int mode = p.getInt("touch_mode", 0);
        if (mode == 2) return null;
        float[] alphas = {0.30f, 0.50f, 0.75f};
        TouchControls t = new TouchControls(activity, p, mode, alphas[Math.max(0, Math.min(2, p.getInt("touch_opacity", 1)))]);
        t.start();
        return t;
    }

    private TouchControls(Activity activity, android.content.SharedPreferences prefs, int mode, float alpha) {
        super(activity);
        this.activity = activity;
        this.prefs = prefs;
        this.vibrate = prefs.getBoolean("touch_vibrate", true);
        loadLayout();
        this.mode = mode;
        this.alpha = alpha;
        ring.setStyle(Paint.Style.STROKE);
        ring.setColor(Color.WHITE);
        text.setColor(Color.WHITE);
        text.setTextAlign(Paint.Align.CENTER);
        text.setTypeface(Typeface.create(Typeface.DEFAULT, Typeface.BOLD));
        small.setColor(0xCCFFFFFF);
        small.setTextAlign(Paint.Align.CENTER);
        setSystemUiVisibility(SYSTEM_UI_FLAG_HIDE_NAVIGATION | SYSTEM_UI_FLAG_FULLSCREEN | SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                | SYSTEM_UI_FLAG_LAYOUT_STABLE | SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION | SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN);
    }

    private void start() {
        InputManager im = (InputManager) activity.getSystemService(Context.INPUT_SERVICE);
        im.registerInputDeviceListener(this, handler);
        handler.post(this::tryAttach);
    }

    /** The virtual pad can only be made once the game has started SDL's gamepad support: retry until then. */
    private void tryAttach() {
        if (attached) return;
        try {
            attached = nativeAttach();
        } catch (UnsatisfiedLinkError e) {
            return;  // libmain without touch support
        }
        if (!attached) {
            handler.postDelayed(this::tryAttach, 500);
            return;
        }
        updateVisibility();
    }

    void stop() {
        handler.removeCallbacksAndMessages(null);
        ((InputManager) activity.getSystemService(Context.INPUT_SERVICE)).unregisterInputDeviceListener(this);
        releaseAll();
        if (added) {
            try { activity.getWindowManager().removeView(this); } catch (RuntimeException ignored) { }
            added = false;
        }
        if (attached) {
            nativeDetach();
            attached = false;
        }
    }

    private static boolean controllerConnected() {
        for (int id : InputDevice.getDeviceIds()) {
            InputDevice d = InputDevice.getDevice(id);
            if (d == null || d.isVirtual()) continue;
            int s = d.getSources();
            if ((s & InputDevice.SOURCE_GAMEPAD) == InputDevice.SOURCE_GAMEPAD
                    || (s & InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK) return true;
        }
        return false;
    }

    private void updateVisibility() {
        if (!attached) return;
        boolean want = mode == 1 || !controllerConnected();
        if (want && !added) {
            if (activity.getWindow().getDecorView().getWindowToken() == null) {
                handler.postDelayed(this::updateVisibility, 300);
                return;
            }
            WindowManager.LayoutParams lp = new WindowManager.LayoutParams(
                    WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.MATCH_PARENT,
                    WindowManager.LayoutParams.TYPE_APPLICATION_PANEL,
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN
                            | WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS | WindowManager.LayoutParams.FLAG_SPLIT_TOUCH,
                    PixelFormat.TRANSLUCENT);
            lp.token = activity.getWindow().getDecorView().getWindowToken();
            lp.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
            try {
                activity.getWindowManager().addView(this, lp);
                added = true;
            } catch (RuntimeException e) {
                handler.postDelayed(this::updateVisibility, 1000);
                return;
            }
        }
        shown = want;
        setVisibility(want ? VISIBLE : GONE);
        if (!want) releaseAll();
    }

    @Override public void onInputDeviceAdded(int id) { updateVisibility(); }
    @Override public void onInputDeviceRemoved(int id) { updateVisibility(); }
    @Override public void onInputDeviceChanged(int id) { updateVisibility(); }

    @Override
    protected void onSizeChanged(int w, int h, int ow, int oh) {
        place();
        stickR = 0.15f * h;
        ring.setStrokeWidth(Math.max(2f, h / 360f));
        float bw = 0.16f * h, bh = 0.09f * h, gap = 0.02f * h, x0 = w / 2f - 2 * bw - 1.5f * gap, y0 = 0.17f * h;
        for (int i = 0; i < 4; ++i) editBar[i] = new RectF(x0 + i * (bw + gap), y0, x0 + i * (bw + gap) + bw, y0 + bh);
    }

    /** Screen positions from the fractional layout. */
    private void place() {
        int w = getWidth(), h = getHeight();
        if (h == 0) return;
        for (Btn b : buttons) {
            b.r = b.fr * h;
            b.cx = b.fromRight ? w - b.fx * h : b.fx * h;
            b.cy = b.fy * h;
        }
        Btn rs = rsBtn();
        rsR = rs.r; rsCx = rs.cx; rsCy = rs.cy;
    }

    private void loadLayout() {
        String saved = prefs.getString("touch_layout", "");
        for (String item : saved.split(";")) {
            String[] kv = item.split(":");
            if (kv.length != 2) continue;
            String[] v = kv[1].split(",");
            try {
                int i = Integer.parseInt(kv[0]);
                if (i < 0 || i >= buttons.length || v.length != 3) continue;
                buttons[i].fx = Float.parseFloat(v[0]);
                buttons[i].fy = Float.parseFloat(v[1]);
                buttons[i].fr = Float.parseFloat(v[2]);
            } catch (NumberFormatException ignored) { }
        }
    }

    private void saveLayout() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < buttons.length; ++i) {
            Btn b = buttons[i];
            if (b.fx == b.dfx && b.fy == b.dfy && b.fr == b.dfr) continue;
            sb.append(i).append(':').append(b.fx).append(',').append(b.fy).append(',').append(b.fr).append(';');
        }
        prefs.edit().putString("touch_layout", sb.toString()).apply();
    }

    // ---- input ----

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        if (!shown) return false;
        int action = e.getActionMasked();
        int idx = e.getActionIndex();
        if (editing) {
            editTouch(e, action, idx);
            invalidate();
            return true;
        }
        switch (action) {
            case MotionEvent.ACTION_DOWN:
            case MotionEvent.ACTION_POINTER_DOWN:
                down(e.getPointerId(idx), e.getX(idx), e.getY(idx));
                break;
            case MotionEvent.ACTION_MOVE:
                for (int i = 0; i < e.getPointerCount(); ++i) move(e.getPointerId(i), e.getX(i), e.getY(i));
                break;
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_POINTER_UP:
                up(e.getPointerId(idx));
                break;
            case MotionEvent.ACTION_CANCEL:
                releaseAll();
                break;
        }
        invalidate();
        return true;
    }

    private Btn hit(float x, float y) {
        Btn best = null;
        float bestD = Float.MAX_VALUE;
        for (Btn b : buttons) {
            if (b.id == RS_PAD) continue;
            float d = (float) Math.hypot(x - b.cx, y - b.cy);
            if (d < b.r * 1.15f && d < bestD) { best = b; bestD = d; }
        }
        return best;
    }

    private void down(int pid, float x, float y) {
        Btn b = hit(x, y);
        if (b != null && b.id == EDIT) {
            releaseAll();
            editing = true;
            editSel = null;
            return;
        }
        if (b != null && b.id == TRIGGER + AX_LT) {  // RUN: a toggle
            runToggled = !runToggled;
            updateLt();
            if (vibrate) performHapticFeedback(android.view.HapticFeedbackConstants.VIRTUAL_KEY);
            return;
        }
        if (b != null) {
            if (b.pointer < 0) press(b, true);
            b.pointer = pid;
            return;
        }
        if (rsPointer < 0 && Math.hypot(x - rsCx, y - rsCy) < rsR * 1.6f) {
            rsPointer = pid;
            rsDownTime = System.currentTimeMillis();
            rsX = x; rsY = y;
            sendRs();
            return;
        }
        if (stickPointer < 0 && x < getWidth() * 0.5f && y > getHeight() * 0.35f) {
            stickPointer = pid;
            stickBaseX = Math.max(stickR, Math.min(x, getWidth() * 0.5f - stickR));
            stickBaseY = Math.max(getHeight() * 0.35f + stickR * 0.5f, Math.min(y, getHeight() - stickR));
            stickX = x; stickY = y;
            sendStick();
        }
    }

    private void move(int pid, float x, float y) {
        if (pid == stickPointer) {
            stickX = x; stickY = y;
            sendStick();
        } else if (pid == rsPointer) {
            rsX = x; rsY = y;
            sendRs();
        } else {
            // Sliding a finger from one button onto another (e.g. STRIKE -> GRAPPLE) switches the press.
            for (Btn b : buttons) {
                if (b.pointer != pid) continue;
                if (Math.hypot(x - b.cx, y - b.cy) > b.r * 1.4f) {
                    Btn other = hit(x, y);
                    press(b, false);
                    b.pointer = -1;
                    if (other != null && other.pointer < 0) { other.pointer = pid; press(other, true); }
                }
                return;
            }
        }
    }

    private void up(int pid) {
        if (pid == stickPointer) {
            stickPointer = -1;
            nativeAxis(AX_LX, 0); nativeAxis(AX_LY, 0);
            setRunning(false);
        } else if (pid == rsPointer) {
            rsPointer = -1;
            boolean tap = System.currentTimeMillis() - rsDownTime < 250 && Math.hypot(rsX - rsCx, rsY - rsCy) < rsR * 0.5f;
            nativeAxis(AX_RX, 0); nativeAxis(AX_RY, 0);
            if (tap) {  // a quick tap clicks the right stick (store a finisher)
                nativeButton(RS, true);
                handler.postDelayed(() -> nativeButton(RS, false), 80);
            }
        }
        for (Btn b : buttons) {
            if (b.pointer == pid) { b.pointer = -1; press(b, false); }
        }
    }

    private void releaseAll() {
        for (Btn b : buttons) if (b.pointer >= 0) { b.pointer = -1; press(b, false); }
        runToggled = false;
        if (stickPointer >= 0) { stickPointer = -1; nativeAxis(AX_LX, 0); nativeAxis(AX_LY, 0); }
        if (rsPointer >= 0) { rsPointer = -1; nativeAxis(AX_RX, 0); nativeAxis(AX_RY, 0); }
        setRunning(false);
        invalidate();
    }

    private void press(Btn b, boolean down) {
        if (!attached) return;
        if (!down) {
            // A press must last a few frames, or the game (which reads the pad once a frame) can miss a quick tap.
            long held = System.currentTimeMillis() - b.downTime;
            if (held < 100) {
                handler.postDelayed(() -> { if (b.pointer < 0) press(b, false); }, 100 - held);
                return;
            }
        } else {
            b.downTime = System.currentTimeMillis();
        }
        if (b.id >= TRIGGER) {
            nativeAxis(b.id - TRIGGER, down ? 1f : 0f);
        } else {
            nativeButton(b.id, down);
        }
        if (down && vibrate) performHapticFeedback(android.view.HapticFeedbackConstants.VIRTUAL_KEY);
    }

    private void sendStick() {
        float dx = (stickX - stickBaseX) / stickR, dy = (stickY - stickBaseY) / stickR;
        float len = (float) Math.hypot(dx, dy);
        if (len > 1f) { dx /= len; dy /= len; }
        if (len < 0.12f) { dx = 0; dy = 0; }  // dead zone
        nativeAxis(AX_LX, dx);
        nativeAxis(AX_LY, dy);
        setRunning(len > 1.25f);  // dragged past the ring: run
    }

    private void setRunning(boolean run) {
        if (run == running) return;
        running = run;
        updateLt();
    }

    /** LT (run) is held while RUN is switched on or the stick is pushed past its ring. */
    private void updateLt() {
        boolean want = runToggled || running;
        if (want == ltDown || !attached) return;
        ltDown = want;
        nativeAxis(AX_LT, want ? 1f : 0f);
        invalidate();
    }

    // ---- layout editor ----

    private void editTouch(MotionEvent e, int action, int idx) {
        float x = e.getX(idx), y = e.getY(idx);
        int pid = e.getPointerId(idx);
        switch (action) {
            case MotionEvent.ACTION_DOWN:
            case MotionEvent.ACTION_POINTER_DOWN: {
                if (editPointer >= 0) return;
                for (int i = 0; i < 4; ++i) {
                    if (!editBar[i].contains(x, y)) continue;
                    if (i < 2 && editSel != null) {
                        editSel.fr = Math.max(0.035f, Math.min(0.2f, editSel.fr * (i == 0 ? 0.9f : 1.1f)));
                    } else if (i == 2) {
                        for (Btn b : buttons) { b.fx = b.dfx; b.fy = b.dfy; b.fr = b.dfr; }
                        editSel = null;
                    } else if (i == 3) {
                        editing = false;
                        editSel = null;
                    }
                    saveLayout();
                    place();
                    return;
                }
                Btn best = null;
                float bestD = Float.MAX_VALUE;
                for (Btn b : buttons) {
                    if (b.id == EDIT) continue;
                    float d = (float) Math.hypot(x - b.cx, y - b.cy);
                    if (d < Math.max(b.r * 1.2f, getHeight() * 0.06f) && d < bestD) { best = b; bestD = d; }
                }
                editSel = best;
                editDrag = best;
                if (best != null) {
                    editPointer = pid;
                    editOffX = best.cx - x;
                    editOffY = best.cy - y;
                }
                break;
            }
            case MotionEvent.ACTION_MOVE:
                if (editDrag == null) return;
                for (int i = 0; i < e.getPointerCount(); ++i) {
                    if (e.getPointerId(i) != editPointer) continue;
                    int w = getWidth(), h = getHeight();
                    float cx = Math.max(editDrag.r, Math.min(w - editDrag.r, e.getX(i) + editOffX));
                    float cy = Math.max(editDrag.r, Math.min(h - editDrag.r, e.getY(i) + editOffY));
                    editDrag.fx = (editDrag.fromRight ? w - cx : cx) / h;
                    editDrag.fy = cy / h;
                    place();
                }
                break;
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_POINTER_UP:
            case MotionEvent.ACTION_CANCEL:
                if (pid == editPointer || action == MotionEvent.ACTION_CANCEL) {
                    editPointer = -1;
                    editDrag = null;
                    saveLayout();
                }
                break;
        }
    }

    private void sendRs() {
        float dx = (rsX - rsCx) / rsR, dy = (rsY - rsCy) / rsR;
        float len = (float) Math.hypot(dx, dy);
        if (len > 1f) { dx /= len; dy /= len; }
        if (len < 0.25f) { dx = 0; dy = 0; }
        nativeAxis(AX_RX, dx);
        nativeAxis(AX_RY, dy);
    }

    // ---- drawing ----

    @Override
    protected void onDraw(Canvas c) {
        if (!shown) return;
        int a = (int) (alpha * 255);
        if (editing) {
            fill.setColor(Color.BLACK);
            fill.setAlpha(110);
            c.drawRect(0, 0, getWidth(), getHeight(), fill);
        }
        for (Btn b : buttons) {
            if (b.id == RS_PAD || (editing && b.id == EDIT)) continue;
            boolean pressed = b.pointer >= 0 || (b.id == TRIGGER + AX_LT && ltDown) || (editing && b == editSel);
            fill.setColor(b.color);
            fill.setAlpha(pressed ? Math.min(255, a + 110) : a);
            c.drawCircle(b.cx, b.cy, b.r, fill);
            ring.setAlpha(pressed ? 255 : Math.min(255, a + 60));
            c.drawCircle(b.cx, b.cy, b.r, ring);
            text.setAlpha(Math.min(255, a + 140));
            float size = b.r * (b.label.length() > 6 ? 0.30f : 0.38f);
            if (b.label.length() == 1) size = b.r * 0.8f;
            text.setTextSize(size);
            float ty = b.sub.isEmpty() ? b.cy + size * 0.35f : b.cy + size * 0.1f;
            c.drawText(b.label, b.cx, ty, text);
            if (!b.sub.isEmpty()) {
                small.setTextSize(Math.max(b.r * 0.22f, 18f));
                small.setAlpha(Math.min(255, a + 100));
                c.drawText(b.sub, b.cx, b.label.length() == 1 ? b.cy + b.r + small.getTextSize() : ty + small.getTextSize() * 1.15f, small);
            }
        }
        // Right stick.
        fill.setColor(GREY);
        fill.setAlpha(a / 2);
        c.drawCircle(rsCx, rsCy, rsR, fill);
        ring.setAlpha(Math.min(255, a + 60));
        c.drawCircle(rsCx, rsCy, rsR, ring);
        float kx = rsCx, ky = rsCy;
        if (rsPointer >= 0) {
            float dx = rsX - rsCx, dy = rsY - rsCy, len = (float) Math.hypot(dx, dy);
            if (len > rsR) { dx *= rsR / len; dy *= rsR / len; }
            kx += dx; ky += dy;
        }
        fill.setColor(Color.WHITE);
        fill.setAlpha(Math.min(255, a + 40));
        c.drawCircle(kx, ky, rsR * 0.45f, fill);
        small.setTextSize(Math.max(rsR * 0.3f, 18f));
        small.setAlpha(Math.min(255, a + 100));
        c.drawText("MOVE OPP. · TAP = STORE", rsCx, rsCy + rsR + small.getTextSize() * 1.1f, small);
        // Movement stick: a hint where it appears, or the stick itself while held.
        float bx = stickPointer >= 0 ? stickBaseX : 0.20f * getHeight() + stickR * 0.4f;
        float by = stickPointer >= 0 ? stickBaseY : getHeight() - stickR * 1.25f;
        fill.setColor(GREY);
        fill.setAlpha(a / 2);
        c.drawCircle(bx, by, stickR, fill);
        ring.setAlpha(Math.min(255, a + 60));
        c.drawCircle(bx, by, stickR, ring);
        ring.setAlpha(running ? 255 : a / 2);
        c.drawCircle(bx, by, stickR * 1.25f, ring);
        float sx = bx, sy = by;
        if (stickPointer >= 0) {
            float dx = stickX - bx, dy = stickY - by, len = (float) Math.hypot(dx, dy);
            float max = stickR * 1.25f;
            if (len > max) { dx *= max / len; dy *= max / len; }
            sx += dx; sy += dy;
        }
        fill.setColor(running ? ORANGE : Color.WHITE);
        fill.setAlpha(Math.min(255, a + 60));
        c.drawCircle(sx, sy, stickR * 0.42f, fill);
        if (stickPointer < 0) {
            small.setTextSize(Math.max(stickR * 0.17f, 18f));
            c.drawText("MOVE · PUSH TO EDGE = RUN", bx, by + stickR * 1.25f + small.getTextSize() * 1.1f, small);
        }
        if (editing) {
            if (editSel != null && editSel.id == RS_PAD) {
                ring.setAlpha(255);
                c.drawCircle(rsCx, rsCy, rsR * 1.1f, ring);
            }
            String[] labels = {"−  SMALLER", "+  BIGGER", "RESET", "DONE"};
            text.setAlpha(255);
            text.setTextSize(getHeight() * 0.032f);
            for (int i = 0; i < 4; ++i) {
                boolean on = i >= 2 || editSel != null;
                fill.setColor(i == 3 ? GREEN : GREY);
                fill.setAlpha(on ? 230 : 90);
                c.drawRoundRect(editBar[i], 16, 16, fill);
                c.drawText(labels[i], editBar[i].centerX(), editBar[i].centerY() + text.getTextSize() * 0.35f, text);
            }
            small.setAlpha(255);
            small.setTextSize(getHeight() * 0.03f);
            c.drawText("Drag a button to move it. Tap one, then − / + to resize.", getWidth() / 2f, editBar[0].top - getHeight() * 0.03f, small);
        }
    }
}
