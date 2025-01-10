#include "stm32f10x.h"                  // Device header
#include "OLED.h"
#include "Serial.h"
#include "string.h"

int main(void)
{
    /*模块初始化*/
    OLED_Init();        // OLED初始化
    Serial_Init();      // 串口初始化

    /*清屏，准备显示内容*/
    OLED_Clear();
    
    while (1)
    {
        if (Serial_RxFlag == 1)  // 如果接收到数据包
        {
            int x[4], y[4]; // 存储坐标数据
            
            /* 解析四点坐标 */
            if (sscanf(Serial_RxPacket, "(%d,%d),(%d,%d),(%d,%d),(%d,%d)",
                       &x[0], &y[0], &x[1], &y[1], &x[2], &y[2], &x[3], &y[3]) == 8)
            {
                /* 显示四点坐标 */
                char buffer[16];
                
                sprintf(buffer, "P1:%d,%d", x[0], y[0]);
                OLED_ShowString(1, 1, "                "); // 清除旧内容
                OLED_ShowString(1, 1, buffer);

                sprintf(buffer, "P2:%d,%d", x[1], y[1]);
                OLED_ShowString(2, 1, "                "); // 清除旧内容
                OLED_ShowString(2, 1, buffer);

                sprintf(buffer, "P3:%d,%d", x[2], y[2]);
                OLED_ShowString(3, 1, "                "); // 清除旧内容
                OLED_ShowString(3, 1, buffer);

                sprintf(buffer, "P4:%d,%d", x[3], y[3]);
                OLED_ShowString(4, 1, "                "); // 清除旧内容
                OLED_ShowString(4, 1, buffer);
            }
            else
            {
                /* 数据解析错误处理 */
                OLED_ShowString(1, 1, "Invalid Packet ");
                OLED_ShowString(2, 1, "                ");
                OLED_ShowString(3, 1, "                ");
                OLED_ShowString(4, 1, "                ");
            }
            
            Serial_RxFlag = 0; // 清除接收标志位
        }
    }
}
