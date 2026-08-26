#ifndef _RM_SERIAL_DRIVER_PACKET_HPP
#define _RM_SERIAL_DRIVER_PACKET_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rm::serial
{
constexpr std::size_t kArmJointCount = 6;
constexpr uint8_t kSendHeader = 0xA5;
constexpr uint8_t kReceiveHeader = 0x5A;
constexpr uint8_t kLegacyEndFrame = 0x4A;
constexpr uint8_t kArmExtensionHeader = 0xB5;
constexpr uint8_t kArmExtensionEndFrame = 0x4B;

struct ArmJointPacket
{
  float position[kArmJointCount] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
} __attribute__((packed));

struct ReceivePacket
{
  uint8_t header = kReceiveHeader;
  uint8_t end_frame = kLegacyEndFrame;
  uint8_t arm_header = kArmExtensionHeader;
  uint8_t arm_state_valid = 0;
  uint32_t arm_sequence = 0;
  ArmJointPacket arm;
  uint8_t arm_end_frame = kArmExtensionEndFrame;
} __attribute__((packed));

struct NavPacket
{
  float linear_x = 0.0;
  float linear_y = 0.0;
  float angular_z = 0.0;
} __attribute__((packed));

struct SendPacket
{
  uint8_t header = kSendHeader;
  NavPacket nav;
  uint8_t end_frame = kLegacyEndFrame;
  uint8_t arm_header = kArmExtensionHeader;
  uint8_t arm_command_enabled = 0;
  uint32_t arm_sequence = 0;
  ArmJointPacket arm;
  uint8_t arm_end_frame = kArmExtensionEndFrame;
} __attribute__((packed));

ReceivePacket FromVector(const std::vector<uint8_t>& data);
std::vector<uint8_t> ToVector(const SendPacket& data, bool include_arm_extension = true);
std::size_t SendPacketSize(bool include_arm_extension);
std::size_t LegacySendPacketSize();
bool HasValidArmReceiveExtension(const ReceivePacket& packet, std::size_t size);
}

#endif
