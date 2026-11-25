package com.toni.wwe13;

import android.os.Bundle;
import org.libsdl.app.SDLActivity;

public class WWE13Activity extends SDLActivity {
    private TouchControls touch;

    @Override
    protected String[] getLibraries() {
        // SDL3 is linked statically into libmain.so.
        return new String[] { "rexruntime", "main" };
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Nothing of the activity window may cover the game's SurfaceView.
        getWindow().setBackgroundDrawable(null);
        // Put the game surface above the activity window so nothing can cover it.
        if (mSurface != null) {
            mSurface.setZOrderOnTop(true);
        }
        // On-screen touch controls (launcher setting; hidden while a controller is connected by default).
        touch = TouchControls.install(this);
    }

    @Override
    protected void onDestroy() {
        if (touch != null) touch.stop();
        touch = null;
        super.onDestroy();
    }

    @Override
    protected void onStop() {
        super.onStop();
        // Leaving the game (home button, switching apps): upload changed saves if save sync is on.
        if (SaveSync.isSetUp(this) && SaveSync.prefs(this).getBoolean("sync_auto", true)) {
            final android.content.Context app = getApplicationContext();
            new Thread(() -> {
                LauncherNative.init(app.getExternalFilesDir(null).getAbsolutePath());
                SaveSync.sync(app, null, true);
            }).start();
        }
    }
}
