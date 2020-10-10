package com.p4f.esp32camai;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Rect;
import android.text.TextPaint;
import android.util.AttributeSet;
import android.view.Gravity;
import android.widget.Button;

public class VerticalButton extends Button{
    final boolean topDown;

    public VerticalButton(Context context, AttributeSet attrs){
        super(context, attrs);
        final int gravity = getGravity();
//        if(Gravity.isVertical(gravity) && (gravity&Gravity.VERTICAL_GRAVITY_MASK) == Gravity.BOTTOM) {
            setGravity((gravity&Gravity.HORIZONTAL_GRAVITY_MASK) | Gravity.TOP);
            topDown = true;
//        }else
//            topDown = false;
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec){
        super.onMeasure(heightMeasureSpec, widthMeasureSpec);
        setMeasuredDimension(getMeasuredHeight(), getMeasuredWidth());
    }

    @SuppressLint("DrawAllocation")
    @Override
    protected void onDraw(Canvas canvas){
        TextPaint textPaint = getPaint();
        textPaint.setColor(getCurrentTextColor());
        textPaint.drawableState = getDrawableState();

        Rect bounds = new Rect();
        textPaint.getTextBounds(getText().toString(), 0, getText().length(), bounds);
        int textWidth= bounds.height();
        int widthLocation =Math.abs( ((getMeasuredWidth()/2)-(textWidth/2)));

        canvas.save();

        if(topDown){
            canvas.translate(getWidth(), 0);
            canvas.rotate(90);
        }else {
            canvas.translate(0, getHeight());
            canvas.rotate(-90);
        }

        canvas.translate(getCompoundPaddingLeft(), widthLocation);
        getLayout().draw(canvas);
        canvas.restore();
    }
}
