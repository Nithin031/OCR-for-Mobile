plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.ocr.core"
    compileSdk = 35

    defaultConfig {
        minSdk = 26
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.exifinterface)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.mlkit.text.recognition)
    implementation(libs.onnxruntime.android)
    implementation(libs.opencv)

    testImplementation(libs.junit)
    androidTestImplementation(libs.junit.ext)
}

// ── Asset copy task ─────────────────────────────────────────────────────────
// Copies Phase-0 model assets (JSON configs + keys file) from the reference
// directory into the Android assets so they are always in sync.
// ONNX files are gitignored in the source; run tools/reference/build_android_assets.py
// to populate them before building Phase 4+.
val copyModels by tasks.registering(Copy::class) {
    description = "Copies Phase-0 model assets into ocr-core/src/main/assets/models/"
    from("$rootDir/tools/reference/android_assets/models")
    into("$projectDir/src/main/assets/models")
}

tasks.named("preBuild") {
    dependsOn(copyModels)
}
